"""Export the ADM (pa-audioic) to ONNX for the browser, and check the export.

Usage (from the repository root):
    python code/python/admonnx.py export --model-dir local/adm-models --out-dir local/adm-onnx
    python code/python/admonnx.py check  --model-dir local/adm-models --out-dir local/adm-onnx AUDIO.wav

IC at noise level t, as pa-audioic computes it (ICCalcHelper.ic with the
expected noise process; model time runs from 1 = clean to 0 = noise, so
s = 1 - t):
    z = s * latents                      (Music2Latent latents, T x 64)
    c = transformer(z)[:-1]              (context of frames 2..T, 256-d)
    x(s) = z[1:]; integrate dx/dt = v(t, x, c) from t = s to 0 with 100 Euler
    steps, accumulating L = sum dt * tr(dv/dx) (Skilling-Hutchinson estimate
    with 4 Rademacher vectors, fixed over the steps)
    IC = -(log N(x(0); 0, I) + L)        (one value per frame 2..T)
Only the diffusion MLP v runs inside the ODE. The export has three graphs:
    encoder.onnx    Music2Latent encoder, STFT representation -> latents
                    (B x 2 x 1024 x F -> B x 64 x F/8; divided by sigma_rescale)
    context.onnx    causal transformer, z (1 x T x 64) -> context (1 x T x 256)
    mlpstep.onnx    one evaluation of v with its trace estimate:
                    x (N x 64), c (N x 256), t (1), e (R x N x 64) ->
                    v (N x 64), tr (N) = mean_r e_r . (dv/dx) e_r
                    (IC: R = 4, N = frames; entropy: R = 1, N = frames x samples)
Entropy at noise level t, as pa-audioic's entr (pred_new): the transformer
context of the clean latents; 32 samples z0 ~ N(0, I) per frame integrated
forward from model time 0 to s = 1 - t (Euler grid of 100 steps over [0, 1]),
one Rademacher vector; log p = log N(z0) - integral of the trace; entropy =
mean of -log p over the samples.
The Jacobian-vector products are written out explicitly (forward-mode), so the
graph has only standard ONNX operators; `check` compares them with torch.func.
The STFT representation, silence trimming, 60-s encoder chunks and the Euler
loop are left to the caller (numpy here, JavaScript in code/web/adm.js);
`check` runs this numpy version against pa-audioic on one audio file and saves
pa-audioic's latents, Rademacher vectors and IC in OUT/test/ for
code/web/test-adm.mjs.

HISTORY
2026-11-20: written (SG) (agent)
2026-11-20: entropy; any number of trace vectors in mlpstep.onnx (SG) (agent)
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from admcompute import (ENTROPY_RUNS, EULER_STEPS, FRAME_SEC, IC_RUNS, SAMPLE_RATE, getweights,  # noqa: E402
                        loadhelper, readaudio)

HOP = 512                  # Music2Latent STFT hop (hparams.hop)
WIN = 4 * HOP              # STFT frame (fac = 4)
DOWN = 8                   # STFT frames per latent frame (2 ** freq_downsample_list.count(0))
ALPHA, BETA = 0.65, 0.34   # normalize_complex
SIGMA_RESCALE = 0.06
CHUNK = (int(SAMPLE_RATE * 60 / HOP) // DOWN) * DOWN   # max_sample_length of encode_audio_inference
LN_EPS = 1e-5
ENT_FRAMES, ENT_SAMPLES = 200, 32   # entropy check: first frames of the test file; Monte Carlo samples (admcompute)


# ---- numpy reference of the browser's preprocessing --------------------------
def trimbounds(audio, top_db=60, frame_length=2048, hop_length=512):
    """First non-silent sample and end of the last non-silent part, as
    librosa.effects.split(audio)[0][0] and [-1][1] (RMS with centre padding)."""
    pad = frame_length // 2
    y = np.pad(audio, (pad, pad))
    nFrames = 1 + (len(y) - frame_length) // hop_length
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(nFrames)[:, None]
    rms = np.sqrt(np.mean(y[idx] ** 2, axis=1))
    db = 10 * np.log10(np.maximum(1e-10, rms ** 2)) - 10 * np.log10(np.maximum(1e-10, rms.max() ** 2))
    loud = np.flatnonzero(db > -top_db)
    if len(loud) == 0:
        return 0, 0
    return min(loud[0] * hop_length, len(audio)), min((loud[-1] + 1) * hop_length, len(audio))


def representation(audio):
    """Encoder input of music2latent (crop, frame, Hann, rfft, normalize_complex),
    float32 array 2 x 1024 x F."""
    n = ((((len(audio) - 3 * HOP) // HOP) // DOWN) * HOP * DOWN) + 3 * HOP
    audio = audio[:n].astype(np.float32)
    nFrames = 1 + (len(audio) - WIN) // HOP
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)   # periodic Hann (torch default)
    frames = audio[np.arange(WIN)[None, :] + HOP * np.arange(nFrames)[:, None]] * window
    X = np.fft.rfft(frames, axis=-1)[:, :2 * HOP].T                  # 1024 x F
    X = BETA * np.abs(X) ** ALPHA * np.exp(1j * np.angle(X))
    return np.stack([X.real, X.imag]).astype(np.float32)


def encode(encSession, audio):
    """Latents (T x 64) as encode_audio_inference: 60-s chunks, the last one
    padded with copies of the representation, then concatenated and cropped."""
    rep = representation(audio)
    F = rep.shape[-1]
    pad = 0
    if F > CHUNK:
        pad = CHUNK - F % CHUNK
        reps = np.concatenate([rep] * (1 + pad // F), axis=-1)[..., :pad]
        rep = np.concatenate([rep, reps], axis=-1)
        rep = np.stack(np.split(rep, rep.shape[-1] // CHUNK, axis=-1))
    else:
        rep = rep[None]
    lat = np.concatenate([encSession.run(None, {"rep": r[None]})[0] for r in rep], axis=0)  # chunks x 64 x L
    lat = np.concatenate(list(lat), axis=-1)
    lat = lat[:, :lat.shape[-1] - pad // DOWN]
    return lat.T                                                       # T x 64


def icfromlatents(ctxSession, mlpSession, latents, level, e):
    """IC (frames 2..T) at noise level `level` with the Euler loop of
    LikelihoodSolver.diffusion_logp; e: R x (T-1) x 64 Rademacher vectors."""
    s = 1.0 - level
    z = (s * latents).astype(np.float32)
    ctx = ctxSession.run(None, {"z": z[None]})[0][0, :-1]
    x = z[1:].astype(np.float64)
    L = np.zeros(len(x))
    ts = np.linspace(s, 0.0, EULER_STEPS + 1)
    for i in range(EULER_STEPS):
        v, tr = mlpSession.run(None, {"x": x.astype(np.float32), "c": ctx,
                                      "t": np.array([ts[i]], np.float32), "e": e})
        dt = ts[i + 1] - ts[i]
        x = x + dt * v
        L = L + dt * tr
    logpz = -0.5 * (x ** 2).sum(-1) - 0.5 * x.shape[-1] * math.log(2 * math.pi)
    return -(logpz + L)


def entfromlatents(ctxSession, mlpSession, latents, levels, z0, e):
    """Entropy (frames 2..T) at each noise level, as pa_audioic's entr
    (ODETransformerModel.pred_new): context from the clean latents; M samples
    z0 (M x (T-1) x 64) integrated forward from model time 0 with 100 Euler
    steps, e: 1 x M x (T-1) x 64; log p = log N(z0) - integral of the trace up
    to 1 - level; entropy = mean over the samples of -log p. The steps beyond
    the largest 1 - level do not change the result and are skipped."""
    M, N = z0.shape[:2]
    ctx = ctxSession.run(None, {"z": latents[None].astype(np.float32)})[0][0, :-1]
    ts = np.linspace(0.0, 1.0, EULER_STEPS + 1)
    stops = {int(np.flatnonzero(np.isclose(ts, 1 - lv))[0]): k for k, lv in enumerate(levels)}
    x = z0.reshape(M * N, -1).astype(np.float64)
    c = np.tile(ctx, (M, 1))
    logpz = (-0.5 * (x ** 2).sum(-1) - 0.5 * x.shape[-1] * math.log(2 * math.pi))
    L = np.zeros(M * N)
    H = np.zeros((len(levels), N))
    for i in range(max(stops) + 1):
        if i in stops:
            H[stops[i]] = (-(logpz - L)).reshape(M, N).mean(0)
        if i == max(stops):
            break
        v, tr = mlpSession.run(None, {"x": x.astype(np.float32), "c": c, "t": np.array([ts[i]], np.float32),
                                      "e": e.reshape(e.shape[0], M * N, -1)})
        dt = ts[i + 1] - ts[i]
        x = x + dt * v
        L = L + dt * tr
    return H


# ---- explicit forward-mode MLP for the export ----------------------------------
def lndual(h, dh):
    mu = h.mean(-1, keepdim=True)
    c = h - mu
    r = torch.rsqrt((c * c).mean(-1, keepdim=True) + LN_EPS)
    dc = dh - dh.mean(-1, keepdim=True)
    return c * r, dc * r - c * r ** 3 * (c * dc).mean(-1, keepdim=True)


class MlpStep(torch.nn.Module):
    """v(t, x, c) of DiffusionMLPContext and tr = mean_r e_r . J e_r."""

    def __init__(self, mlp):
        super().__init__()
        self.m = mlp

    def forward(self, x, c, t, e):
        m = self.m
        cond = m.emb_proj(m.time_emb(t * 2.0 - 1.0))                    # 1 x 256
        h = m.linear_input(torch.cat([x, c], -1))                        # N x 256
        dh = e @ m.linear_input.weight[:, :x.shape[-1]].T                # R x N x 256 (c is constant)
        for layer in m.layers:
            n, dn = lndual(h, dh)
            g = 1 + layer.norm.scale_proj(cond)
            a, da = n * g, dn * g
            u1, u2 = layer.ff.to_mlp(a).chunk(2, -1)
            du1, du2 = (da @ layer.ff.to_mlp.weight.T).chunk(2, -1)
            sg = torch.sigmoid(u1)
            s, ds = u1 * sg, sg * (1 + u1 * (1 - sg)) * du1
            h = h + layer.ff.to_out(s * u2)
            dh = dh + (ds * u2 + s * du2) @ layer.ff.to_out.weight.T
        n, dn = lndual(h, dh)
        g = 1 + m.norm_output.scale_proj(cond)
        a = n * g + m.norm_output.shift_proj(cond)
        v = m.linear_output(a)
        dv = (dn * g) @ m.linear_output.weight.T
        return v, (dv * e).sum(-1).mean(0)


class Context(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.b = backbone

    def forward(self, z):
        return self.b(z)


class Encoder(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, rep):
        return self.enc(rep) / SIGMA_RESCALE


def export(helper, outDir):
    outDir.mkdir(parents=True, exist_ok=True)
    model, mlp = helper.model, helper.model.diff_mlp
    T, N, R = 64, 63, IC_RUNS
    Tdim = torch.export.Dim("T", min=2, max=4600)
    Ndim, Rdim = torch.export.Dim("N", min=1, max=2 ** 20), torch.export.Dim("R", min=1, max=64)  # N: frames x samples
    Fdim = torch.export.Dim("F8", min=1, max=CHUNK // DOWN)
    torch.onnx.export(Encoder(helper.encdec.gen.encoder).eval(), (torch.randn(1, 2, 2 * HOP, 8 * 16),),
                      outDir / "encoder.onnx", dynamo=True, input_names=["rep"], output_names=["lat"],
                      dynamic_shapes={"rep": {3: DOWN * Fdim}}, external_data=False)
    torch.onnx.export(Context(model.backbone).eval(), (torch.randn(1, T, 64),), outDir / "context.onnx",
                      dynamo=True, input_names=["z"], output_names=["ctx"],
                      dynamic_shapes={"z": {1: Tdim}}, external_data=False)
    torch.onnx.export(MlpStep(mlp).eval(), (torch.randn(N, 64), torch.randn(N, 256), torch.tensor([0.3]),
                                            torch.randint(2, (R, N, 64)).float() * 2 - 1),
                      outDir / "mlpstep.onnx", dynamo=True, input_names=["x", "c", "t", "e"],
                      output_names=["v", "tr"],
                      dynamic_shapes={"x": {0: Ndim}, "c": {0: Ndim}, "t": None, "e": {0: Rdim, 1: Ndim}},
                      external_data=False)
    writemeta(outDir)
    for f in sorted(outDir.glob("*.onnx")):
        print(f"{f.name}: {f.stat().st_size / 2**20:.1f} MB")


def writemeta(outDir):
    """adm.json: settings and a version (hash of the graphs), which the page
    uses to name its cache, so a new export replaces cached graphs."""
    import hashlib

    h = hashlib.sha256()
    for n in ("encoder", "context", "mlpstep"):
        h.update((outDir / f"{n}.onnx").read_bytes())
    meta = {"version": h.hexdigest()[:12], "frame_sec": FRAME_SEC, "sample_rate": SAMPLE_RATE, "hop": HOP,
            "win": WIN, "down": DOWN, "alpha": ALPHA, "beta": BETA, "euler_steps": EULER_STEPS,
            "ic_runs": IC_RUNS, "entropy_runs": ENTROPY_RUNS, "max_frames": 4600}
    (outDir / "adm.json").write_text(json.dumps(meta, indent=2))


def check(helper, outDir, fname, levels=(0.0, 0.6), seed=0):
    import onnxruntime as ort
    from pa_audioic.audioic import IntegrationParams

    torch.manual_seed(1)
    mlp = helper.model.diff_mlp
    x, c, t = torch.randn(20, 64), torch.randn(20, 256), torch.tensor([0.37])
    e = torch.randint(2, (IC_RUNS, 20, 64)).float() * 2 - 1
    v, tr = MlpStep(mlp)(x, c, t, e)
    f = lambda xx: mlp(t, xx, c)   # t of shape (1,): one time embedding for all frames
    jv = torch.stack([torch.func.jvp(f, (x,), (er,))[1] for er in e])
    print(f"forward-mode trace vs torch.func.jvp: max |diff| v {(v - f(x)).abs().max():.2e}, "
          f"tr {(tr - (jv * e).sum(-1).mean(0)).abs().max():.2e}")

    opts = ort.SessionOptions()
    sess = {n: ort.InferenceSession(str(outDir / f"{n}.onnx"), opts, providers=["CPUExecutionProvider"])
            for n in ("encoder", "context", "mlpstep")}
    audio, _ = readaudio(fname)
    head, tail, latRef = helper.encode_m2l(audio, silence_extractor="music")
    a, b = trimbounds(audio)
    print(f"trim: numpy {a}..{b}, frames head {a // 4096}, tail {math.ceil((len(audio) - b) / 4096)}; "
          f"pa-audioic head {head}, tail {tail}")
    lat = encode(sess["encoder"], audio[a:b])
    latRef = latRef.numpy()
    print(f"latents: {lat.shape} vs {latRef.shape}, max |diff| {np.abs(lat - latRef).max():.2e}, "
          f"r {np.corrcoef(lat.ravel(), latRef.ravel())[0, 1]:.6f}")

    torch.manual_seed(seed)
    icRef = helper.ic([head], [tail], [torch.from_numpy(latRef)], noise_levels=list(levels),
                      integration_params=IntegrationParams(n_runs=IC_RUNS, solver="euler",
                                                           solver_kwargs={"n_steps": EULER_STEPS}))[0]
    # Test data for code/web/test-adm.mjs: pa-audioic's latents, Rademacher
    # vectors and IC (float32/float64 little-endian), and the settings
    testDir = outDir / "test"
    testDir.mkdir(exist_ok=True)
    latRef.astype("<f4").tofile(testDir / "latents.f32")
    meta = {"audio": str(fname), "frames": int(len(latRef)), "head": int(head), "tail": int(tail),
            "trim": [int(a), int(b)], "levels": list(levels)}
    torch.manual_seed(seed)   # the same Rademacher vectors as pa-audioic: one draw per level, in order
    for k, level in enumerate(levels):
        e = (torch.randint(2, size=(IC_RUNS, 1, len(latRef) - 1, 64)).float() * 2 - 1)[:, 0].numpy()
        ic = icfromlatents(sess["context"], sess["mlpstep"], latRef, level, e)
        ref = icRef[k][~np.isnan(icRef[k])]
        e.astype("<f4").tofile(testDir / f"e_{level}.f32")
        ref.astype("<f8").tofile(testDir / f"ic_{level}.f64")
        print(f"IC t = {level}: ONNX vs pa-audioic, max |diff| {np.abs(ic - ref).max():.3f}, "
              f"r {np.corrcoef(ic, ref)[0, 1]:.6f}, mean {ic.mean():.2f} vs {ref.mean():.2f}")
    # Entropy on the first ENT_FRAMES latent frames (pa-audioic's entr is slow on a CPU)
    lat = latRef[:ENT_FRAMES]
    torch.manual_seed(seed)
    hRef = helper.entr([0], [0], [torch.from_numpy(lat)], noise_levels=list(levels), monte_carlo_samples=ENT_SAMPLES,
                       integration_params=IntegrationParams(n_runs=ENTROPY_RUNS, solver="euler",
                                                            solver_kwargs={"n_steps": EULER_STEPS}))[0]
    torch.manual_seed(seed)   # the draws of pred_new: samples, then Rademacher vectors
    z0 = helper.model.ode.latent_cond(64).sample((ENT_SAMPLES, len(lat) - 1)).numpy()
    e = (torch.randint(2, size=(ENTROPY_RUNS, ENT_SAMPLES, len(lat) - 1, 64)).float() * 2 - 1).numpy()
    H = entfromlatents(sess["context"], sess["mlpstep"], lat, levels, z0, e)
    z0.astype("<f4").tofile(testDir / "ent_z0.f32")
    e.astype("<f4").tofile(testDir / "ent_e.f32")
    for k, level in enumerate(levels):
        hRef[k].astype("<f8").tofile(testDir / f"ent_{level}.f64")
        print(f"entropy t = {level} ({ENT_FRAMES} frames, {ENT_SAMPLES} samples): ONNX vs pa-audioic, "
              f"max |diff| {np.abs(H[k] - hRef[k]).max():.3f}, r {np.corrcoef(H[k], hRef[k])[0, 1]:.6f}, "
              f"mean {H[k].mean():.2f} vs {hRef[k].mean():.2f}")
    meta.update(ent_frames=ENT_FRAMES, ent_samples=ENT_SAMPLES)
    (testDir / "test.json").write_text(json.dumps(meta, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["export", "check"])
    p.add_argument("audio", nargs="?")
    p.add_argument("--model-dir", default="local/adm-models")
    p.add_argument("--out-dir", default="local/adm-onnx")
    args = p.parse_args()
    helper = loadhelper(getweights(args.model_dir), "cpu")
    helper.model.eval()
    if args.command == "export":
        export(helper, Path(args.out_dir))
    else:
        check(helper, Path(args.out_dir), args.audio)


if __name__ == "__main__":
    main()
