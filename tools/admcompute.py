"""Compute frame-wise IC and entropy of audio files with the ADM (pa-audioic).

Usage:
    python admcompute.py --out-dir OUT [--list FILES.txt] [FILE ...]

Each audio file gives OUT/<stem>.csv with the columns
    Frame, Time, IC_<t>, Entropy_<t>   (one IC and one entropy column per t)
and the run gives OUT/admcompute_<timestamp>.json (versions, settings, files).

Model: the perceptually aligned rectified flow of Bjare et al. (2026, EUSIPCO),
pa-audioic branch `entropy` (github.com/CPJKU/pa-audioic, c74dbe9), weights
from Hugging Face CPJKU/pa-audioic at a fixed revision (checked by SHA-256).

Settings (as in MEX, config/baseline_pa_audioic_session.yaml, except the
number of Monte Carlo samples for entropy):
    IC:      Euler solver, 100 steps, 4 Skilling-Hutchinson runs
    entropy: Euler solver, 100 steps, 1 Skilling-Hutchinson run,
             32 Monte Carlo samples, with the variance reduction below
    noise levels t = 0.0 and 0.6 (0 = clean, 1 = pure noise)

Frames: Music2Latent frames last 4096/44100 s (about 10.8 Hz). Row j of the
CSV is latent frame `Frame` (1, 2, ...), starting at `Time` seconds. IC is
-log p(frame | earlier frames); entropy is the expected IC of that frame given
the same context, i.e., before the frame is heard. Leading and trailing
silence (librosa.effects.split, 60 dB) are NaN; the audio is cut at the first
non-silent sample, so after a silent start `Time` is accurate to within one
frame. MEX plots IC at `Frame` but
entropy at `Frame - 1` (time_offset_frame 1 and 0); here both use `Frame`.

Entropy estimator: pa-audioic estimates the entropy of frame n at level t as
the mean over samples z0 ~ N(0, I) of -log p(x_t) = -log N(z0) + int tr(dv/dx),
integrated from z0 to x_t. -log N(z0) = |z0|^2/2 + D/2 log(2 pi) varies
between samples (SD about 5.6 nats for D = 64) but not between frames, and
its mean is known; at t >= 0.3 it is nearly all the variance of the estimate.
Here it is replaced by its expectation, D/2 (1 + log 2 pi), so the estimate is
H(N(0, I)) + mean of int tr(dv/dx) (the change-of-variables identity; same
expected value, same draws). Frame-wise reliability with 32 samples at t = 0.6
rose from .39-.83 to .97-.996 over ten test files (local/entvar/, 2026-11-23).
--plain-entropy gives pa-audioic's (and MEX's) estimator.

Random numbers (trace estimator, entropy samples) come from torch's global
generator, which is reseeded with --seed before each file and each metric,
so a file gives the same values whatever else is in the run.

HISTORY
2026-11-22: 128 Monte Carlo samples for entropy by default (was 32, as MEX) (SG) (agent)
2026-11-23: entropy with the expectation of -log N(z0) (variance reduction);
            32 Monte Carlo samples by default again (SG) (agent)
"""

import argparse
import hashlib
import json
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from contextlib import contextmanager

SAMPLE_RATE = 44100
FRAME_SEC = 4096 / SAMPLE_RATE
MAX_FRAMES = 4600  # max_seq_len of pa_rectflow.pt
EULER_STEPS = 100
IC_RUNS = 4  # Skilling-Hutchinson trace-estimator runs
ENTROPY_RUNS = 1
HF_REPO = "CPJKU/pa-audioic"
HF_REVISION = "917beb5fe491cb2af0414db7d4b2cf425e5deeb6"  # 2025-11-17
HF_SHA256 = {
    "pa_rectflow.pt": "0d76f5b23e745ed410de3be686d3adeafcf022aa296a44df7677c4587050edcb",
    "music2latent.pt": "370086c4d9807b17c7c5dcd6196a9dc43a54c726e37a7dea6a2fd48fc80a4d5f",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("files", nargs="*", help="audio files")
    p.add_argument("--list", help="text file with one audio path per line")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-dir", default=str(Path.home() / ".cache" / "adm-models"),
                   help="where the pinned weights are downloaded and kept")
    p.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.6])
    p.add_argument("--mc-samples", type=int, default=32, help="Monte Carlo samples for entropy (MEX: 32)")
    p.add_argument("--plain-entropy", action="store_true",
                   help="pa-audioic's estimator, without the variance reduction (as MEX)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true", help="recompute files whose CSV exists")
    return p.parse_args()


def sha256(fname):
    h = hashlib.sha256()
    with open(fname, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def getweights(modelDir):
    """Download the pinned weights (once) and check their SHA-256."""
    from huggingface_hub import hf_hub_download

    paths = {}
    for fname, expected in HF_SHA256.items():
        path = Path(modelDir) / fname
        if not path.exists():
            hf_hub_download(HF_REPO, fname, revision=HF_REVISION, local_dir=modelDir)
        if sha256(path) != expected:
            raise RuntimeError(f"{path}: SHA-256 differs from revision {HF_REVISION[:7]}")
        paths[fname] = str(path)
    return paths


def loadhelper(weights, device):
    """ICCalcHelper on the pinned weights, without the packages' own downloads."""
    import music2latent.inference as m2l

    # EncoderDecoder downloads the latest music2latent.pt into site-packages
    # unless it exists there; point it at the pinned file instead.
    m2l.download_model = lambda: None
    m2l.load_path_inference_default = weights["music2latent.pt"]
    from pa_audioic.audioic import ICCalcHelper

    return ICCalcHelper(device=device, ckpt=weights["pa_rectflow.pt"])


@contextmanager
def expectedbaselogprob(ode):
    """Within the block, log N(z0) of the base distribution of `ode` is its
    expectation, -D/2 (1 + log 2 pi), for every z0; sampling is unchanged.
    pa-audioic's entropy (pred_new, forward) is then H(N(0, I)) + int tr."""
    original = ode.latent_cond

    class ExpectedLogProb(torch.distributions.Independent):
        def log_prob(self, value):
            d = self.event_shape[0]
            return torch.full(value.shape[:-1], -0.5 * d * (1 + np.log(2 * np.pi)), dtype=value.dtype)

    def latentcond(n):
        base = original(n)
        return ExpectedLogProb(base.base_dist, base.reinterpreted_batch_ndims)

    ode.latent_cond = latentcond
    try:
        yield
    finally:
        del ode.latent_cond  # back to the class method


def readaudio(fname):
    """Mono float32 at 44.1 kHz (pa-audioic takes arrays as 44.1 kHz)."""
    audio, sr = sf.read(fname, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio, sr


def vcsinfo(dist):
    """Commit of a package installed from git, or its version."""
    try:
        d = metadata.distribution(dist)
        url = d.read_text("direct_url.json")
        if url:
            vcs = json.loads(url).get("vcs_info", {})
            if "commit_id" in vcs:
                return {"version": d.version, "commit": vcs["commit_id"],
                        "requested_revision": vcs.get("requested_revision")}
        return {"version": d.version}
    except metadata.PackageNotFoundError:
        return None


def main():
    args = parse_args()
    fnames = list(args.files)
    if args.list:
        fnames += [l.strip() for l in Path(args.list).read_text().splitlines()
                   if l.strip() and not l.startswith("#")]
    if not fnames:
        raise SystemExit("no audio files given")
    weights = getweights(args.model_dir)

    from pa_audioic.audioic import IntegrationParams

    outDir = Path(args.out_dir)
    outDir.mkdir(parents=True, exist_ok=True)
    helper = loadhelper(weights, args.device)
    solverKwargs = {"n_steps": EULER_STEPS}
    icParams = IntegrationParams(n_runs=IC_RUNS, solver="euler", solver_kwargs=solverKwargs)
    entrParams = IntegrationParams(n_runs=ENTROPY_RUNS, solver="euler", solver_kwargs=solverKwargs)
    levels = args.noise_levels

    log = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "settings": {k: v for k, v in vars(args).items() if k not in ("files", "list")}
                    | {"euler_steps": EULER_STEPS, "ic_runs": IC_RUNS, "entropy_runs": ENTROPY_RUNS},
        "frame_sec": FRAME_SEC,
        "weights": {"repo": HF_REPO, "revision": HF_REVISION, "sha256": HF_SHA256},
        "packages": {d: vcsinfo(d) for d in ("pa_audioic", "music2latent", "torch", "librosa", "soundfile")},
        "device_name": torch.cuda.get_device_name() if args.device.startswith("cuda") else args.device,
        "files": [],
    }
    logName = outDir / f"admcompute_{datetime.now():%Y%m%d-%H%M%S}.json"

    for i, fname in enumerate(fnames, 1):
        csvName = outDir / f"{Path(fname).stem}.csv"
        if csvName.exists() and not args.overwrite:
            print(f"[{i}/{len(fnames)}] skip (exists): {csvName}")
            continue
        tic = time.time()
        audio, sr = readaudio(fname)
        head, trail, latents = helper.encode_m2l(audio, silence_extractor="music")
        if latents.shape[0] > MAX_FRAMES:
            raise ValueError(f"{fname}: {latents.shape[0]} frames > {MAX_FRAMES} ({MAX_FRAMES * FRAME_SEC:.0f} s)")

        torch.manual_seed(args.seed)
        ic = helper.ic([head], [trail], [latents], noise_levels=levels, integration_params=icParams)[0]
        torch.manual_seed(args.seed)
        if args.plain_entropy:
            entr = helper.entr([head], [trail], [latents], noise_levels=levels,
                               integration_params=entrParams, monte_carlo_samples=args.mc_samples)[0]
        else:
            with expectedbaselogprob(helper.model.ode):
                entr = helper.entr([head], [trail], [latents], noise_levels=levels,
                                   integration_params=entrParams, monte_carlo_samples=args.mc_samples)[0]
        assert ic.shape == entr.shape, (ic.shape, entr.shape)

        frame = np.arange(1, ic.shape[1] + 1)
        table = np.column_stack([frame, frame * FRAME_SEC, ic.T, entr.T])
        header = ",".join(["Frame", "Time"] + [f"IC_{t}" for t in levels] + [f"Entropy_{t}" for t in levels])
        np.savetxt(csvName, table, delimiter=",", header=header, comments="",
                   fmt=["%d", "%.6f"] + ["%.6f"] * (table.shape[1] - 2))

        sec = time.time() - tic
        log["files"].append({"audio": str(fname), "csv": str(csvName), "sample_rate": sr,
                             "duration_sec": len(audio) / SAMPLE_RATE, "frames": int(ic.shape[1]),
                             "nan_head": int(head), "nan_tail": int(trail), "seconds": round(sec, 1)})
        logName.write_text(json.dumps(log, indent=2))
        print(f"[{i}/{len(fnames)}] {sec:.0f} s: {csvName}")

    log["finished"] = datetime.now().isoformat(timespec="seconds")
    logName.write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
