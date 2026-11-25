// Plot of IC and entropy over time for one file: two stacked panels (IC;
// entropy if computed) on a shared time axis, one line per noise level
// (categorical slots of the reference palette, validated on this page's light
// and dark surfaces), a crosshair that snaps to the nearest frame in both
// panels, and a tooltip with every value at that time. NaN frames (trimmed
// silence) leave gaps. Plain SVG, no library.
//
// HISTORY
// 2026-11-21: written (SG) (agent)

const NS = "http://www.w3.org/2000/svg";
const W = 860, PANEL_H = 170, M = { left: 52, right: 64, top: 22, bottom: 8 }, AXIS_H = 34;

function el(name, attrs = {}, parent = null) {
  const e = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.append(e);
  return e;
}

// "Nice" tick values covering [lo, hi]
function ticks(lo, hi, n) {
  const span = hi - lo || 1, raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return { values: out, step };
}
const fmtTime = (s) => (s >= 60 ? `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}` : `${+s.toFixed(1)} s`);

// data: { title, frameSec, head, levels, ics: [Float64Array], ents: [Float64Array] | null }
export function renderPlot(container, data) {
  const { frameSec, head, levels, ics, ents } = data;
  const panels = [{ name: "IC", unit: "nats", series: ics }];
  if (ents) panels.push({ name: "Entropy", unit: "nats", series: ents });
  const N = ics[0].length, nRows = head + N + (data.tail || 0);
  const time = (j) => (head + j + 1) * frameSec;            // Time of IC value j (as the CSV)
  const tMax = nRows * frameSec;
  const x = (t) => M.left + (t / tMax) * (W - M.left - M.right);
  const H = panels.length * (PANEL_H + M.top + M.bottom) + AXIS_H;

  container.replaceChildren();
  const fig = document.createElement("figure");
  fig.className = "viz-root";
  const cap = document.createElement("figcaption");
  cap.textContent = data.title;
  fig.append(cap);
  if (levels.length > 1) {                                   // legend for >= 2 series
    const lg = document.createElement("div");
    lg.className = "legend";
    levels.forEach((t, k) => {
      const item = document.createElement("span"), key = document.createElement("span");
      key.className = "key"; key.style.background = `var(--series-${k + 1})`;
      item.append(key, document.createTextNode(`t = ${t}`));
      lg.append(item);
    });
    fig.append(lg);
  }
  const wrap = document.createElement("div");
  wrap.className = "plotwrap";
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
                          "aria-label": `${panels.map((p) => p.name).join(" and ")} over time for ${data.title}` });
  wrap.append(svg);
  fig.append(wrap);
  container.append(fig);

  const geo = [];                                            // per panel: y scale
  panels.forEach((p, i) => {
    const top = i * (PANEL_H + M.top + M.bottom) + M.top, bottom = top + PANEL_H;
    let lo = Infinity, hi = -Infinity;
    for (const s of p.series) for (const v of s) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const pad = (hi - lo) * 0.05 || 1;
    lo -= pad; hi += pad;
    const y = (v) => bottom - ((v - lo) / (hi - lo)) * PANEL_H;
    geo.push({ y, top, bottom });
    const title = el("text", { x: M.left, y: top - 8, class: "ptitle" }, svg);
    title.textContent = `${p.name} (${p.unit})`;
    for (const v of ticks(lo, hi, 4).values) {
      el("line", { x1: M.left, x2: W - M.right, y1: y(v), y2: y(v), class: "grid" }, svg);
      const lab = el("text", { x: M.left - 6, y: y(v) + 4, class: "tick", "text-anchor": "end" }, svg);
      lab.textContent = String(v);
    }
    const ends = [];
    p.series.forEach((s, k) => {
      let d = "", pen = false;
      for (let j = 0; j < s.length; j++) {
        if (!Number.isFinite(s[j])) { pen = false; continue; }
        d += `${pen ? "L" : "M"}${x(time(j)).toFixed(1)},${y(s[j]).toFixed(1)}`;
        pen = true;
      }
      el("path", { d, class: "line", stroke: `var(--series-${k + 1})` }, svg);
      let j = s.length - 1;
      while (j > 0 && !Number.isFinite(s[j])) j--;
      if (Number.isFinite(s[j])) ends.push({ k, x: x(time(j)) + 6, y: y(s[j]) + 4 });
    });
    if (levels.length > 1) {                                 // direct labels at the line ends, kept 14 px apart
      ends.sort((a, b) => a.y - b.y);
      for (let i = 1; i < ends.length; i++) {
        if (Math.abs(ends[i].x - ends[i - 1].x) < 60 && ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
      }
      for (const e of ends) el("text", { x: e.x, y: e.y, class: "tick" }, svg).textContent = `t = ${levels[e.k]}`;
    }
  });
  const axisY = H - AXIS_H + 4;
  for (const t of ticks(0, tMax, 8).values) {
    el("line", { x1: x(t), x2: x(t), y1: axisY - 4, y2: axisY, class: "axis" }, svg);
    const lab = el("text", { x: x(t), y: axisY + 14, class: "tick", "text-anchor": "middle" }, svg);
    lab.textContent = fmtTime(t);
  }
  const xl = el("text", { x: W - M.right, y: axisY + 28, class: "tick", "text-anchor": "end" }, svg);
  xl.textContent = "Time";

  // Crosshair and tooltip, snapping to the nearest frame
  const hair = el("line", { y1: geo[0].top, y2: geo.at(-1).bottom, class: "hair", visibility: "hidden" }, svg);
  const dots = panels.map((p, i) => p.series.map((_, k) => el("circle", { r: 4, class: "dot", fill: `var(--series-${k + 1})`, visibility: "hidden" }, svg)));
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.hidden = true;
  wrap.append(tip);
  const hit = el("rect", { x: M.left, y: 0, width: W - M.left - M.right, height: H - AXIS_H, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (ev) => {
    const r = svg.getBoundingClientRect(), sx = ((ev.clientX - r.left) / r.width) * W;
    const t = ((sx - M.left) / (W - M.left - M.right)) * tMax;
    const j = Math.max(0, Math.min(N - 1, Math.round(t / frameSec) - head - 1));
    const xp = x(time(j));
    hair.setAttribute("x1", xp); hair.setAttribute("x2", xp); hair.setAttribute("visibility", "visible");
    tip.replaceChildren();
    const h = document.createElement("div");
    h.className = "tiptime";
    h.textContent = `${time(j).toFixed(2)} s (frame ${head + j + 1})`;
    tip.append(h);
    panels.forEach((p, i) => p.series.forEach((s, k) => {
      const ok = Number.isFinite(s[j]), dot = dots[i][k];
      dot.setAttribute("visibility", ok ? "visible" : "hidden");
      if (ok) { dot.setAttribute("cx", xp); dot.setAttribute("cy", geo[i].y(s[j])); }
      const row = document.createElement("div"), key = document.createElement("span"), val = document.createElement("strong");
      key.className = "lkey"; key.style.background = `var(--series-${k + 1})`;
      val.textContent = ok ? s[j].toFixed(2) : "–";
      row.append(key, val, document.createTextNode(` ${p.name}${levels.length > 1 ? `, t = ${levels[k]}` : ""}`));
      tip.append(row);
    }));
    tip.hidden = false;
    const px = (xp / W) * r.width;
    tip.style.left = `${px + 12 + 180 > r.width ? px - 12 - tip.offsetWidth : px + 12}px`;
    tip.style.top = "8px";
  });
  hit.addEventListener("pointerleave", () => {
    hair.setAttribute("visibility", "hidden");
    dots.flat().forEach((d) => d.setAttribute("visibility", "hidden"));
    tip.hidden = true;
  });
}
