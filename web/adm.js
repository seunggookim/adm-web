// IC of the ADM (pa-audioic, perceptually aligned rectified flow; Bjare et al.,
// 2026) from audio, with the ONNX graphs of code/python/admonnx.py. Shared by
// the page (index.html, onnxruntime-web on WebGPU) and the Node.js check
// (test-adm.mjs). Mirrors admonnx.py's numpy reference, which reproduces
// pa-audioic's IC (r = 1.000000 on a Cheung et al. stimulus):
//   audio (mono, 44.1 kHz) -> silence trimmed (librosa.effects.split, 60 dB)
//   -> STFT representation (Music2Latent: Hann 2048, hop 512, beta |X|^alpha)
//   -> encoder in 30-s windows inside pa-audioic's 60-s chunks -> latents (T x 64, 4096 samples per frame)
//   -> z = (1 - t) latents -> transformer context -> 100 Euler steps of the
//   diffusion MLP with a 4-vector Skilling-Hutchinson trace estimate
//   -> IC = -(log N(x(0); 0, I) + integral of the trace), frames 2..T.
// Entropy: Monte Carlo samples integrated forward (computeEntropy), with
// -log N(z0) replaced by its expectation (variance reduction; admcompute.py).
// The random numbers come from a seeded generator, so values differ from
// pa-audioic's (torch's generator) by the estimators' noise only.
//
// HISTORY
// 2026-11-20: written (SG) (agent)
// 2026-11-20: entropy (SG) (agent)
// 2026-11-22: length check before encoding, with an explanation (SG) (agent)
// 2026-11-23: encoder in 30-s windows inside pa-audioic's 60-s chunks (SG) (agent)
// 2026-11-23: entropy with the expectation of -log N(z0); 32 samples (SG) (agent)

export const SAMPLE_RATE = 44100, HOP = 512, WIN = 2048, DOWN = 8, BINS = 1024;
export const ALPHA = 0.65, BETA = 0.34;
export const FRAME_SAMPLES = HOP * DOWN, FRAME_SEC = FRAME_SAMPLES / SAMPLE_RATE;
export const STEPS = 100, RUNS = 4, DIM = 64, CTX_DIM = 256, MAX_FRAMES = 4600;

// ---- Silence (librosa.effects.split: RMS of 2048-sample frames every 512,
// centre-padded, in dB relative to the loudest frame; first and last edge) ----
export function trimBounds(x, topDb = 60, frameLength = 2048, hopLength = 512) {
  const pad = frameLength / 2, n = x.length + 2 * pad;
  const cum = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    const j = i - pad, v = j >= 0 && j < x.length ? x[j] : 0;
    cum[i + 1] = cum[i] + v * v;
  }
  const nFrames = 1 + Math.floor((n - frameLength) / hopLength);
  const power = new Float64Array(nFrames);
  let maxP = 0;
  for (let f = 0; f < nFrames; f++) {
    const o = f * hopLength;
    power[f] = (cum[o + frameLength] - cum[o]) / frameLength;   // rms ** 2
    if (power[f] > maxP) maxP = power[f];
  }
  const ref = 10 * Math.log10(Math.max(1e-10, maxP));
  let first = -1, last = -1;
  for (let f = 0; f < nFrames; f++) {
    if (10 * Math.log10(Math.max(1e-10, power[f])) - ref > -topDb) { if (first < 0) first = f; last = f; }
  }
  if (first < 0) return [0, 0];
  return [Math.min(first * hopLength, x.length), Math.min((last + 1) * hopLength, x.length)];
}

// ---- STFT representation of the Music2Latent encoder ----------------------
let fftTables = null;
function fftInit() {
  const n = WIN, bits = Math.log2(n), rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r; }
  const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos(-2 * Math.PI * i / n); sin[i] = Math.sin(-2 * Math.PI * i / n); }
  const win = new Float64Array(n);
  for (let i = 0; i < n; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n);   // periodic Hann
  fftTables = { rev, cos, sin, win };
}
function fft(re, im) {   // in place, radix 2, length WIN
  const { rev, cos, sin } = fftTables, n = re.length;
  for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let size = 2; size <= n; size *= 2) {
    const half = size / 2, step = n / size;
    for (let s = 0; s < n; s += size) {
      for (let k = 0; k < half; k++) {
        const c = cos[k * step], d = sin[k * step], a = s + k, b = a + half;
        const tr = re[b] * c - im[b] * d, ti = re[b] * d + im[b] * c;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
  }
}

// Returns { data: Float32Array 2 x 1024 x F (channel, bin, frame), F } (F a multiple of 8)
export function representation(x) {
  if (!fftTables) fftInit();
  const n = Math.floor(Math.floor((x.length - 3 * HOP) / HOP) / DOWN) * HOP * DOWN + 3 * HOP;
  if (n < WIN) throw new Error("audio too short (needs at least about 0.2 s of sound)");
  const F = 1 + (n - WIN) / HOP;
  const data = new Float32Array(2 * BINS * F);
  const re = new Float64Array(WIN), im = new Float64Array(WIN);
  for (let f = 0; f < F; f++) {
    const o = f * HOP;
    for (let i = 0; i < WIN; i++) { re[i] = Math.fround(x[o + i]) * fftTables.win[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k < BINS; k++) {
      const mag = Math.hypot(re[k], im[k]);
      const g = mag > 0 ? BETA * Math.pow(mag, ALPHA) / mag : 0;
      data[k * F + f] = re[k] * g;
      data[(BINS + k) * F + f] = im[k] * g;
    }
  }
  return { data, F };
}

// ---- Encoder in 30-s windows inside pa-audioic's 60-s chunks --------------
// pa-audioic encodes in chunks of PA_CHUNK STFT frames (60 s), the last one
// padded with copies of the representation from its start. A 60-s chunk does
// not fit in browser memory (WebAssembly: out of memory; 6.6 GB on the CPU),
// so each chunk is encoded in windows of ENC_WINDOW frames (29.9 s): the
// central ENC_CHUNK frames are kept and ENC_MARGIN frames on each side are
// context only; at a chunk's edges the window is shifted inwards so that it
// stays 29.9 s long. The encoder's instance and group normalisations take
// their statistics over the whole input, so the latents differ slightly from
// pa-audioic's; keeping pa-audioic's chunk boundaries and padding matters
// most (largest latent difference on a Cheung et al. stimulus: 1.21 with
// 15-s chunks and 3-s margins, 0.09 with these windows).
// runEncoder(Float32Array 2 x 1024 x Fc, Fc) -> Float32Array 64 x Fc/8
export const PA_CHUNK = Math.floor(Math.floor(SAMPLE_RATE * 60 / HOP) / DOWN) * DOWN;   // 5160
export const ENC_CHUNK = 1552, ENC_MARGIN = 512, ENC_WINDOW = ENC_CHUNK + 2 * ENC_MARGIN;
export async function encode(runEncoder, x) {
  const { data, F } = representation(x);
  const T = F / DOWN, lat = new Float32Array(T * DIM);
  const Fp = F > PA_CHUNK ? F + PA_CHUNK - (F % PA_CHUNK) : F;   // padded as pa-audioic: frame j is j % F
  for (let c0 = 0; c0 < F; c0 += PA_CHUNK) {
    const Lc = Math.min(PA_CHUNK, Fp - c0);
    for (let s = 0; s < Lc && c0 + s < F; s += ENC_CHUNK) {
      let lo = s - ENC_MARGIN, hi = s + ENC_CHUNK + ENC_MARGIN;
      if (Lc >= ENC_WINDOW) {
        if (lo < 0) [lo, hi] = [0, ENC_WINDOW];
        if (hi > Lc) [lo, hi] = [Lc - ENC_WINDOW, Lc];
      }
      lo = Math.max(0, lo); hi = Math.min(Lc, hi);
      const Fc = hi - lo, chunk = new Float32Array(2 * BINS * Fc);
      for (let row = 0; row < 2 * BINS; row++) {
        for (let j = 0; j < Fc; j++) chunk[row * Fc + j] = data[row * F + ((c0 + lo + j) % F)];
      }
      const out = await runEncoder(chunk, Fc), L = Fc / DOWN;
      const first = (s - lo) / DOWN, keep = Math.min(ENC_CHUNK, Lc - s, F - c0 - s) / DOWN;
      for (let l = 0; l < keep; l++) {
        for (let d = 0; d < DIM; d++) lat[((c0 + s) / DOWN + l) * DIM + d] = out[d * L + first + l];
      }
    }
  }
  return { lat, T };
}

// ---- Rademacher vectors (mulberry32) --------------------------------------
function rademacherUniforms(seed, n) {   // n uniforms in [0, 1)
  let a = seed >>> 0;
  const u = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    u[i] = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  return u;
}
export function rademacher(seed, n) {
  return Float32Array.from(rademacherUniforms(seed, n), (u) => (u < 0.5 ? -1 : 1));
}

// Standard normal samples (Box-Muller on mulberry32 uniforms)
export function gaussian(seed, n) {
  const u = rademacherUniforms(seed, n + (n % 2)), z = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const r = Math.sqrt(-2 * Math.log(1 - u[i])), a = 2 * Math.PI * u[i + 1];
    z[i] = r * Math.cos(a);
    if (i + 1 < n) z[i + 1] = r * Math.sin(a);
  }
  return z;
}

// ---- IC at one noise level (LikelihoodSolver.diffusion_logp, Euler) --------
// runContext(Float32Array 1 x T x 64, T) -> Float32Array T x 256
// runMlp(x N x 64, c N x 256, N, t, e R x N x 64) -> { v: N x 64, tr: N }
// e: Float32Array RUNS x (T-1) x 64. Returns Float64Array of length T-1.
export async function computeIC(runContext, runMlp, lat, T, level, e, onStep = () => {}) {
  const s = 1 - level, N = T - 1;
  const z = new Float32Array(T * DIM);
  for (let i = 0; i < z.length; i++) z[i] = s * lat[i];
  const ctx = (await runContext(z, T)).slice(0, N * CTX_DIM);
  const x = new Float64Array(N * DIM), L = new Float64Array(N), xf = new Float32Array(N * DIM);
  for (let i = 0; i < x.length; i++) x[i] = z[DIM + i];
  const step = -s / STEPS, ts = Array.from({ length: STEPS + 1 }, (_, i) => (i === STEPS ? 0 : s + i * step));  // np.linspace(s, 0, 101)
  for (let i = 0; i < STEPS; i++) {
    for (let k = 0; k < x.length; k++) xf[k] = x[k];
    const { v, tr } = await runMlp(xf, ctx, N, Math.fround(ts[i]), e);
    const dt = ts[i + 1] - ts[i];
    for (let k = 0; k < x.length; k++) x[k] += dt * v[k];
    for (let n = 0; n < N; n++) L[n] += dt * tr[n];
    onStep(i + 1);
  }
  const ic = new Float64Array(N), logNorm = 0.5 * DIM * Math.log(2 * Math.PI);
  for (let n = 0; n < N; n++) {
    let ss = 0;
    for (let d = 0; d < DIM; d++) ss += x[n * DIM + d] ** 2;
    ic[n] = -(-0.5 * ss - logNorm + L[n]);
  }
  return ic;
}

// ---- Entropy (pa_audioic entr: ODETransformerModel.pred_new, Euler forward) --
// Context from the clean latents; M samples z0 per frame integrated forward
// from model time 0 on the grid of 100 steps over [0, 1]; log p = log N(z0)
// - integral of the trace; entropy at level t = mean of -log p at step
// (1 - t) x 100 (later steps are skipped). By default (plain = false) log N(z0)
// is replaced by its expectation, -D/2 (1 + log 2 pi): the entropy is then
// H(N(0, I)) + mean integral of the trace, with the same expected value and
// much less noise between samples (pa-audioic and MEX: plain = true). Samples are run in batches of at
// most maxRows rows (frames x samples). z0, e: Float32Array M x (T-1) x 64.
// Returns one Float64Array of length T-1 per level.
export async function computeEntropy(runContext, runMlp, lat, T, levels, M, z0, e, onStep = () => {}, maxRows = 16384, plain = false) {
  const N = T - 1, stops = new Map();
  levels.forEach((t, k) => {
    const i = Math.round((1 - t) * STEPS);
    if (Math.abs((1 - t) * STEPS - i) > 1e-6) throw new Error(`entropy needs noise levels on a 0.01 grid (got ${t})`);
    stops.set(i, k);
  });
  const last = Math.max(...stops.keys());
  const ctx = (await runContext(lat, T)).slice(0, N * CTX_DIM);
  const H = levels.map(() => new Float64Array(N)), logNorm = 0.5 * DIM * Math.log(2 * Math.PI);
  const per = Math.max(1, Math.floor(maxRows / N)), nBatches = Math.ceil(M / per);
  let done = 0;
  for (let m0 = 0; m0 < M; m0 += per) {
    const m = Math.min(per, M - m0), rows = m * N;
    const x = Float64Array.from(z0.subarray(m0 * N * DIM, (m0 + m) * N * DIM)), xf = new Float32Array(rows * DIM);
    const eb = e.subarray(m0 * N * DIM, (m0 + m) * N * DIM), L = new Float64Array(rows), logpz = new Float64Array(rows);
    const c = new Float32Array(rows * CTX_DIM);
    for (let j = 0; j < m; j++) c.set(ctx, j * N * CTX_DIM);
    for (let r = 0; r < rows; r++) {
      let ss = 0;
      for (let d = 0; d < DIM; d++) ss += x[r * DIM + d] ** 2;
      logpz[r] = plain ? -0.5 * ss - logNorm : -0.5 * DIM - logNorm;
    }
    for (let i = 0; i <= last; i++) {
      if (stops.has(i)) { const h = H[stops.get(i)]; for (let r = 0; r < rows; r++) h[r % N] += -(logpz[r] - L[r]) / M; }
      if (i === last) break;
      for (let k = 0; k < x.length; k++) xf[k] = x[k];
      const { v, tr } = await runMlp(xf, c, rows, Math.fround(i * (1 / STEPS)), eb);   // np.linspace(0, 1, 101)
      const dt = (i + 1 === STEPS ? 1 : (i + 1) * (1 / STEPS)) - i * (1 / STEPS);
      for (let k = 0; k < x.length; k++) x[k] += dt * v[k];
      for (let r = 0; r < rows; r++) L[r] += dt * tr[r];
      onStep(++done / (nBatches * last));
    }
  }
  return H;
}

// ---- CSV as admcompute.py: Frame, Time, IC_<t>, Entropy_<t>; NaN rows for the trimmed silence
export function toCSV(levels, ics, ents, head, tail) {
  const name = (t) => (Number.isInteger(t) ? t.toFixed(1) : String(t));
  const cols = ics.concat(ents || []), N = cols[0].length;
  const rows = ["Frame,Time," + levels.map((t) => "IC_" + name(t)).concat(ents ? levels.map((t) => "Entropy_" + name(t)) : []).join(",")];
  for (let r = 0; r < head + N + tail; r++) {
    const j = r - head, vals = cols.map((col) => (j >= 0 && j < N ? col[j].toFixed(6) : "nan"));
    rows.push(`${r + 1},${((r + 1) * FRAME_SEC).toFixed(6)},${vals.join(",")}`);
  }
  return rows.join("\n") + "\n";
}

// ---- Length: the transformer's context holds at most MAX_FRAMES frames ----
export const MAX_SEC = MAX_FRAMES * FRAME_SEC;   // 427 s (7.1 min)
export function tooLong(frames) {
  return `too long: ${(frames * FRAME_SEC / 60).toFixed(1)} min of sound (${frames} frames); the model's context holds at most ` +
    `${MAX_FRAMES} frames (${(MAX_SEC / 60).toFixed(1)} min). Cut it into excerpts of at most 7 min (each excerpt starts afresh, ` +
    `with higher IC at its beginning).`;
}
// Frames after silence trimming, before encoding (as encode: F / 8 latent frames)
export function countFrames(audio) {
  const [a, b] = trimBounds(audio), n = b - a;
  return n < WIN ? 0 : Math.floor(Math.floor((n - 3 * HOP) / HOP) / DOWN);
}

// ---- Full pipeline for one mono 44.1-kHz signal ---------------------------
// options: { seed, entropy (bool), samples (Monte Carlo samples for entropy),
//            plainEntropy (bool; pa-audioic's estimator without variance reduction) }
export async function analyse(runners, audio, levels, options, onProgress = () => {}) {
  const { seed = 0, entropy = true, samples = 32, plainEntropy = false } = options;
  const [a, b] = trimBounds(audio);
  if (b <= a) throw new Error("the file is silent");
  if (Math.floor(Math.floor((b - a - 3 * HOP) / HOP) / DOWN) > MAX_FRAMES) {   // checked before the costly steps
    throw new Error(tooLong(Math.floor(Math.floor((b - a - 3 * HOP) / HOP) / DOWN)));
  }
  const head = Math.floor(a / FRAME_SAMPLES), tail = Math.ceil((audio.length - b) / FRAME_SAMPLES);
  onProgress("Encoding", 0);
  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
  const sec = (t) => +((now() - t) / 1000).toFixed(2), timing = {};
  let t0 = now();
  const { lat, T } = await encode(runners.encoder, audio.subarray(a, b));
  timing.encode_sec = sec(t0);
  if (T > MAX_FRAMES) throw new Error(tooLong(T));
  if (T < 2) throw new Error("too short");
  const N = T - 1, e = rademacher(seed, RUNS * N * DIM), ics = [];
  t0 = now();
  for (const [k, level] of levels.entries()) {
    ics.push(await computeIC(runners.context, runners.mlp, lat, T, level, e,
      (i) => onProgress(`IC at t = ${level}`, (k * STEPS + i) / (levels.length * STEPS))));
  }
  timing.ic_sec = sec(t0);
  let ents = null;
  if (entropy) {
    t0 = now();
    const z0 = gaussian(seed + 1, samples * N * DIM), e1 = rademacher(seed + 2, samples * N * DIM);
    ents = await computeEntropy(runners.context, runners.mlp, lat, T, levels, samples, z0, e1,
      (p) => onProgress("Entropy", p), undefined, plainEntropy);
    timing.entropy_sec = sec(t0);
  }
  return { csv: toCSV(levels, ics, ents, head, tail), frames: T, head, tail, ics, ents, timing };
}

// ---- onnxruntime glue (browser or Node.js) ---------------------------------
export async function makeRunners(ort, loadModel, providers) {
  const opts = { executionProviders: providers, graphOptimizationLevel: "all" };
  const sessions = [];   // one at a time: the WebGPU backend cannot create sessions concurrently
  for (const n of ["encoder", "context", "mlpstep"]) sessions.push(await ort.InferenceSession.create(await loadModel(n + ".onnx"), opts));
  const [enc, ctx, mlp] = sessions;
  const T = (data, dims) => new ort.Tensor("float32", data, dims);
  return {
    encoder: async (data, F) => (await enc.run({ rep: T(data, [1, 2, BINS, F]) })).lat.data,
    context: async (z, n) => (await ctx.run({ z: T(z, [1, n, DIM]) })).ctx.data,
    mlp: async (x, c, N, t, e) => {
      const out = await mlp.run({ x: T(x, [N, DIM]), c: T(c, [N, CTX_DIM]), t: T(Float32Array.of(t), [1]),
                                  e: T(e, [e.length / (N * DIM), N, DIM]) });   // RUNS for IC, 1 for entropy
      return { v: out.v.data, tr: out.tr.data };
    },
  };
}
