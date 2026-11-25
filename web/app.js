// Page logic of index.html: loads the ONNX graphs (cached in the browser),
// decodes audio files locally and computes IC and entropy with adm.js, one
// file at a time. Files, or folders (searched recursively for audio), are
// dropped or chosen; results are offered per file, as one ZIP (CSVs in the
// folder structure of the input, plus adm-web-log.json), and, where the
// browser allows (Chrome, Edge), written to a chosen folder as they finish.
// Audio is never uploaded: the only network requests are for the ONNX
// Runtime library and the model files.
//
// HISTORY
// 2026-11-20: written (SG) (agent)
// 2026-11-20: entropy (SG) (agent)
// 2026-11-20: folders, batch output (ZIP, output folder) and log (SG) (agent)
// 2026-11-21: models from models/ next to the page, for GitHub Pages (SG) (agent)
// 2026-11-21: plot of IC and entropy per file (plot.js) (SG) (agent)
// 2026-11-22: early length check (SG) (agent)
// 2026-11-23: entropy estimator in the log (agent)

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs";
import * as adm from "./adm.js";
import { renderPlot } from "./plot.js";

const ORT_VERSION = "1.31.1";
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const MODEL_BASE = new URL(new URLSearchParams(location.search).get("models") || "models/", location.href);   // models/ next to the page (locally: a link to local/adm-onnx)
const AUDIO = /\.(wav|wave|mp3|flac|m4a|aac|mp4|ogg|oga|opus|aif|aiff|aifc|caf|webm)$/i;
const $ = (id) => document.getElementById(id);
let runners = null, modelMeta = {}, backend = "", queue = Promise.resolve();
let outDir = null;                       // FileSystemDirectoryHandle (Save to folder)
const done = [];                         // { csvPath, csv } of finished files
const batch = { total: 0, finished: 0, errors: 0 };
const log = { tool: "ADM in the browser (code/web/index.html)", files: [] };

function status(text, cls = "") { $("status").textContent = text; $("status").className = cls; }

// ---- Model -------------------------------------------------------------------
// The graphs are cached in the browser under the version in adm.json (a hash of
// the graphs, written by admonnx.py export); older versions are deleted.
let cacheName = null;
async function openCache() {
  try {
    modelMeta = await (await fetch(new URL("adm.json", MODEL_BASE), { cache: "no-store" })).json();
    cacheName = "adm-onnx-" + modelMeta.version;
    for (const k of await caches.keys()) if (k.startsWith("adm-onnx-") && k !== cacheName) await caches.delete(k);
  } catch { cacheName = null; /* no adm.json or no Cache Storage: always download */ }
}

async function loadModel(name) {
  const url = new URL(name, MODEL_BASE).href;
  let cache = null;
  try { if (cacheName) cache = await caches.open(cacheName); } catch { /* no Cache Storage (e.g., private window) */ }
  let resp = cache ? await cache.match(url) : null;
  if (!resp) {
    resp = await fetch(url);
    if (!resp.ok) throw new Error(`${name}: HTTP ${resp.status} from ${url}`);
    if (cache) await cache.put(url, resp.clone());
  }
  return new Uint8Array(await resp.arrayBuffer());
}

async function init() {
  const providers = "gpu" in navigator && (await navigator.gpu.requestAdapter()) ? ["webgpu", "wasm"] : ["wasm"];
  status(`Loading the model (about 130 MB the first time; then cached)…`);
  await openCache();
  const t0 = performance.now();
  runners = await adm.makeRunners(ort, loadModel, providers);
  backend = providers[0];
  const name = backend === "webgpu" ? "WebGPU (GPU)" : "WebAssembly (CPU; slower)";
  status(`Ready: ${name}, model loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s.`, "ok");
  for (const id of ["drop", "pickFiles", "pickFolder"]) $(id).disabled = false;
  $("drop").classList.remove("disabled");
}

// ---- Input -------------------------------------------------------------------
// Decode to mono float32 at 44.1 kHz (the browser resamples other rates)
async function decode(file) {
  const ctx = new OfflineAudioContext(1, 1, adm.SAMPLE_RATE);
  const buf = await ctx.decodeAudioData(await file.arrayBuffer());
  const x = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const d = buf.getChannelData(c);
    for (let i = 0; i < d.length; i++) x[i] += d[i];
  }
  for (let i = 0; i < x.length; i++) x[i] /= buf.numberOfChannels;
  return { x, duration: buf.duration };
}

// Dropped items: files and folders (recursively); entries must be taken
// synchronously in the drop event, before any await.
async function filesFromDrop(dataTransfer) {
  const entries = [...dataTransfer.items].map((i) => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
  const out = [];
  async function walk(entry, prefix) {
    if (entry.isFile) {
      out.push({ file: await new Promise((res, rej) => entry.file(res, rej)), path: prefix + entry.name });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      for (let items; (items = await new Promise((res, rej) => reader.readEntries(res, rej))).length;) {
        for (const e of items) await walk(e, prefix + entry.name + "/");
      }
    }
  }
  for (const e of entries) await walk(e, "");
  return out;
}

function parseLevels() {
  const levels = $("levels").value.split(/[,\s]+/).filter(Boolean).map(Number);
  if (!levels.length || levels.some((t) => !(t >= 0 && t < 1))) throw new Error("noise levels must be numbers in [0, 1)");
  return levels;
}

function settings() {
  return { levels: parseLevels(), entropy: $("entropy").checked, samples: Math.max(1, Math.round(Number($("samples").value))),
           seed: Number($("seed").value) >>> 0 };
}

// ---- Output ------------------------------------------------------------------
async function writeToFolder(dir, path, text) {
  const parts = path.split("/");
  for (const p of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(p, { create: true });
  const w = await (await dir.getFileHandle(parts.at(-1), { create: true })).createWritable();
  await w.write(text);
  await w.close();
}

const CRC = new Uint32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
function crc32(b) { let c = 0xFFFFFFFF; for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; }

// ZIP without compression (method 0), UTF-8 names
function makeZip(files) {
  const enc = new TextEncoder(), parts = [], central = [];
  let offset = 0;
  for (const { path, text } of files) {
    const name = enc.encode(path), data = enc.encode(text), crc = crc32(data);
    const head = (sig, extra) => {
      const h = new DataView(new ArrayBuffer(extra ? 46 : 30));
      let p = 0;
      const u32 = (v) => { h.setUint32(p, v, true); p += 4; }, u16 = (v) => { h.setUint16(p, v, true); p += 2; };
      u32(sig); if (extra) u16(20); u16(20); u16(0x0800); u16(0); u16(0); u16(0x21);
      u32(crc); u32(data.length); u32(data.length); u16(name.length); u16(0);
      if (extra) { u16(0); u16(0); u16(0); u32(0); u32(offset); }
      return new Uint8Array(h.buffer);
    };
    const local = head(0x04034b50, false);
    parts.push(local, name, data);
    central.push(head(0x02014b50, true), name);
    offset += local.length + name.length + data.length;
  }
  const cdSize = central.reduce((s, b) => s + b.length, 0), end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true); end.setUint16(10, files.length, true);
  end.setUint32(12, cdSize, true); end.setUint32(16, offset, true);
  return new Blob([...parts, ...central, new Uint8Array(end.buffer)], { type: "application/zip" });
}

function logText() { return JSON.stringify(log, null, 2) + "\n"; }

function updateBatch() {
  const { total, finished, errors } = batch;
  $("batch").textContent = total ? `${finished} of ${total} files done${errors ? `, ${errors} with errors` : ""}` +
    (outDir ? ` · saving to “${outDir.name}”` : "") : "";
  $("zipBtn").disabled = done.length === 0;
}

// ---- Analysis ----------------------------------------------------------------
function addRow(path) {
  const tr = document.createElement("tr");
  tr.innerHTML = `<td></td><td class="info">Waiting…</td><td><progress max="1" value="0"></progress></td><td class="out"></td>`;
  tr.cells[0].textContent = path;
  $("results").append(tr);
  return { info: tr.querySelector(".info"), bar: tr.querySelector("progress"), out: tr.querySelector(".out") };
}

async function analyseFile({ file, path }, row) {
  const t0 = performance.now(), csvPath = path.replace(/\.[^./]+$/, "") + ".csv";
  const record = { path, csv: csvPath };
  try {
    const opts = settings();
    if (!log.settings) {
      log.started = new Date().toISOString();
      log.settings = { noise_levels: opts.levels, entropy: opts.entropy, mc_samples: opts.samples, entropy_estimator: "expected log N(z0)", seed: opts.seed,
                       euler_steps: adm.STEPS, ic_runs: adm.RUNS, entropy_runs: 1,
                       encoder_chunk_sec: +(adm.PA_CHUNK * adm.HOP / adm.SAMPLE_RATE).toFixed(3),
                       encoder_window_sec: +(adm.ENC_WINDOW * adm.HOP / adm.SAMPLE_RATE).toFixed(3),
                       encoder_kept_sec: +(adm.ENC_CHUNK * adm.HOP / adm.SAMPLE_RATE).toFixed(3),
                       encoder_margin_sec: +(adm.ENC_MARGIN * adm.HOP / adm.SAMPLE_RATE).toFixed(3) };
      log.model_version = modelMeta.version || null;
      log.onnxruntime_web = ORT_VERSION;
      log.backend = backend;
      log.user_agent = navigator.userAgent;
    }
    row.info.textContent = "Decoding…";
    const { x, duration } = await decode(file);
    if (duration > adm.MAX_SEC + 60) throw new Error(adm.tooLong(Math.round(duration / adm.FRAME_SEC)));   // clearly too long, even after trimming silence
    const res = await adm.analyse(runners, x, opts.levels, opts, (what, p) => {
      row.info.textContent = `${duration.toFixed(1)} s · ${what}…`; row.bar.value = p;
    });
    const sec = (performance.now() - t0) / 1000;
    row.bar.value = 1;
    row.info.textContent = `${duration.toFixed(1)} s · ${res.frames} frames · encoding ${res.timing.encode_sec.toFixed(1)} s · total ${sec.toFixed(1)} s`;
    Object.assign(record, { duration_sec: +duration.toFixed(3), frames: res.frames, nan_head: res.head, nan_tail: res.tail,
                            seconds: +sec.toFixed(1), timing: res.timing, settings: opts });
    done.push({ csvPath, csv: res.csv });
    const plotData = { title: path, frameSec: adm.FRAME_SEC, head: res.head, tail: res.tail, levels: opts.levels,
                       ics: res.ics, ents: res.ents };
    const btn = document.createElement("button");
    btn.textContent = "Plot";
    btn.className = "small";
    btn.addEventListener("click", () => { renderPlot($("plot"), plotData); $("plot").scrollIntoView({ behavior: "smooth", block: "nearest" }); });
    row.out.append(btn, " ");
    if (!$("plot").hasChildNodes()) renderPlot($("plot"), plotData);   // the first result is shown at once
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([res.csv], { type: "text/csv" }));
    a.download = csvPath.split("/").at(-1);
    a.textContent = "CSV";
    row.out.append(a);
    if (outDir) await writeToFolder(outDir, csvPath, res.csv);
  } catch (e) {
    record.error = String(e.message || e);
    row.info.textContent = "Error: " + record.error;
    row.info.classList.add("bad");
    batch.errors++;
  }
  log.files.push(record);
  if (outDir) await writeToFolder(outDir, "adm-web-log.json", logText()).catch(() => {});
  batch.finished++;
  updateBatch();
}

function addFiles(items) {
  if (!runners) return;
  const audio = items.filter(({ path }) => AUDIO.test(path)).sort((a, b) => a.path.localeCompare(b.path));
  const skipped = items.length - audio.length;
  if (skipped) status(`Skipped ${skipped} file(s) that are not audio (by extension).`);
  batch.total += audio.length;
  updateBatch();
  for (const item of audio) {
    const row = addRow(item.path);
    queue = queue.then(() => analyseFile(item, row));   // one file at a time
  }
}

// ---- Wiring ------------------------------------------------------------------
const drop = $("drop");
const fromInput = (files) => [...files].map((f) => ({ file: f, path: f.webkitRelativePath || f.name }));
$("pickFiles").addEventListener("click", () => $("fileInput").click());
$("pickFolder").addEventListener("click", () => $("folderInput").click());
for (const id of ["fileInput", "folderInput"]) {
  $(id).addEventListener("change", (e) => { addFiles(fromInput(e.target.files)); e.target.value = ""; });
}
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", async (e) => {
  e.preventDefault(); drop.classList.remove("over");
  if (runners) addFiles(await filesFromDrop(e.dataTransfer));
});
$("zipBtn").addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(makeZip([...done.map((d) => ({ path: d.csvPath, text: d.csv })),
                                        { path: "adm-web-log.json", text: logText() }]));
  a.download = "adm-results.zip";
  a.click();
});
if ("showDirectoryPicker" in window) {
  $("folderBtn").hidden = false;
  $("folderBtn").addEventListener("click", async () => {
    try {
      outDir = await window.showDirectoryPicker({ mode: "readwrite" });
      for (const d of done) await writeToFolder(outDir, d.csvPath, d.csv);   // results so far
      if (log.files.length) await writeToFolder(outDir, "adm-web-log.json", logText());
      updateBatch();
    } catch (e) { if (e.name !== "AbortError") status("Could not use that folder: " + e.message, "bad"); }
  });
}
init().catch((e) => status("Could not load the model: " + (e.message || e), "bad"));
