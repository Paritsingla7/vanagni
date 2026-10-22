"use strict";

// Vanagni demo frontend: plain JS, no build step. Every view reads a
// pre-computed API endpoint; nothing here scores or trains.

const SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
const $ = (id) => document.getElementById(id);
const fmt = (x, d = 3) => (x == null ? "–" : Number(x).toFixed(d));
const ci = (r) => (r ? `${fmt(r[0])} – ${fmt(r[1])}` : "");

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);  // append(string) is a text node, never HTML
  return node;
}

async function getJSON(url) {
  const res = await fetch(url);
  const body = await res.json();
  if (!res.ok) throw new Error(body.message || `${res.status}`);
  return body;
}

function showError(target, err) {
  target.replaceChildren(el("p", { class: "status error" }, `Could not load: ${err.message}`));
}

// ---- tooltip (textContent only: labels are untrusted data) -------------------------
const tip = $("tooltip");
function showTip(evt, title, lines) {
  tip.replaceChildren(el("strong", {}, title), ...lines.map((l) => el("div", {}, l)));
  tip.hidden = false;
  tip.style.left = `${evt.clientX + 14}px`;
  tip.style.top = `${evt.clientY + 14}px`;
}
const hideTip = () => { tip.hidden = true; };

// ---- routing -------------------------------------------------------------------------
const SCREENS = { about: "screen-about", map: "screen-map", ablation: "screen-ablation",
                  methodology: "screen-methodology" };
const loaded = {};
function route() {
  const name = (location.hash || "#about").slice(1);
  const key = SCREENS[name] ? name : "about";
  for (const [n, id] of Object.entries(SCREENS)) $(id).hidden = n !== key;
  document.querySelectorAll(".topbar a").forEach((a) =>
    a.classList.toggle("active", a.getAttribute("href") === `#${key}`));
  if (!loaded[key]) {
    loaded[key] = true;
    ({ map: initMap, ablation: initAblation, methodology: initMethodology }[key] || (() => {}))();
  } else if (key === "map" && mapState.map) {
    mapState.map.invalidateSize();
  }
}
window.addEventListener("hashchange", route);

// ---- 1. risk map -----------------------------------------------------------------------
const mapState = { map: null, layer: null, fires: null, data: null, ranges: [], req: 0 };

function riskColor(r, max) {
  const i = Math.min(SEQ.length - 1, Math.floor((r / max) * SEQ.length));
  return SEQ[Math.max(0, i)];
}

async function initMap() {
  const status = $("map-status");
  let index;
  try { index = await getJSON("/api/dates"); } catch (e) { showError(status.parentElement, e); return; }
  mapState.ranges = index.ranges;

  mapState.map = L.map("map").setView([30.1, 79.2], 7);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    { maxZoom: 12, attribution: "© OpenStreetMap contributors" }).addTo(mapState.map);

  const seasonSel = $("season"), dateInp = $("date");
  for (const r of mapState.ranges) seasonSel.append(el("option", { value: r.season }, String(r.season)));
  const setRange = () => {
    const r = mapState.ranges.find((x) => String(x.season) === seasonSel.value);
    dateInp.min = r.start; dateInp.max = r.end;   // only dates with a prediction are selectable
    if (!dateInp.value || dateInp.value < r.start || dateInp.value > r.end) {
      dateInp.value = r.start;
    }
  };
  seasonSel.addEventListener("change", () => { setRange(); loadDate(); });
  dateInp.addEventListener("change", loadDate);
  $("show-fires").addEventListener("change", draw);
  seasonSel.value = mapState.ranges[mapState.ranges.length - 1].season;
  setRange();
  loadDate();
}

async function loadDate() {
  const status = $("map-status");
  const ticket = ++mapState.req;
  status.className = "status";
  status.textContent = "Loading…";             // the map keeps its last render meanwhile
  try {
    const data = await getJSON(`/api/risk?date=${$("date").value}`);
    if (ticket !== mapState.req) return;        // a newer request superseded this one
    mapState.data = data;
    status.textContent = "";
    draw();
  } catch (e) {
    if (ticket !== mapState.req) return;
    status.className = "status error";
    status.textContent = e.message;
  }
}

function draw() {
  const data = mapState.data;
  if (!data) return;
  const max = Math.max(...data.features.map((f) => f.properties.risk), 1e-6);
  const showFires = $("show-fires").checked;
  if (mapState.layer) mapState.layer.remove();
  if (mapState.fires) mapState.fires.remove();

  mapState.layer = L.geoJSON(data, {
    style: (f) => ({ fillColor: riskColor(f.properties.risk, max), fillOpacity: 0.75,
                     weight: 0.3, color: "#898781" }),
    onEachFeature: (f, layer) => {
      layer.on("mousemove", (e) => showTip(e.originalEvent, `Cell ${f.properties.cell_id}`, [
        `Predicted risk ${(f.properties.risk * 100).toFixed(1)}%`,
        `Fire next day: ${f.properties.fire_next_day ? "yes" : "no"}`]));
      layer.on("mouseout", hideTip);
    },
  }).addTo(mapState.map);

  if (showFires) {
    const burned = data.features.filter((f) => f.properties.fire_next_day);
    mapState.fires = L.layerGroup(burned.map((f) => {
      const ring = f.geometry.coordinates[0];
      const lat = ring.reduce((s, p) => s + p[1], 0) / ring.length;
      const lon = ring.reduce((s, p) => s + p[0], 0) / ring.length;
      return L.circleMarker([lat, lon], { radius: 5, color: "#fcfcfb", weight: 2,
                                          fillColor: "#eb6834", fillOpacity: 1 });
    })).addTo(mapState.map);
  }
  drawLegend(max, showFires, data);
}

function drawLegend(max, showFires, data) {
  const burned = data.features.filter((f) => f.properties.fire_next_day).length;
  const swatches = el("span", { class: "swatches" },
    ...SEQ.map((c) => el("span", { style: `background:${c}` })));
  const ramp = el("span", { class: "ramp" }, "Low", swatches, `High (${(max * 100).toFixed(0)}%)`);
  const parts = [ramp, el("span", {}, `${data.features.length} cells · ${burned} burned next day`)];
  if (showFires) parts.push(el("span", { class: "ramp" }, el("span", { class: "dot" }),
                                "Actual fire next day"));
  $("legend").replaceChildren(...parts);
}

// ---- 2. ablation -------------------------------------------------------------------------
const STAGE_NAMES = { "1": "Persistence", "fwi": "Fire Weather Index (operational baseline)",
                      "2": "+ Fire history", "3": "+ Weather",
                      "4a": "+ Fuel & terrain", "4b": "+ Human proximity",
                      "4b-weather": "Full model minus weather" };

async function initAblation() {
  const body = $("ablation-body");
  let a;
  try { a = await getJSON("/api/ablation"); } catch (e) { showError(body, e); return; }

  const rows = a.stages.map((s) => {
    const m = s.mean;
    const seasons = el("details", {}, el("summary", {}, "per season"),
      ...s.per_season.map((r) => el("div", { class: "ci" },
        `${r.season}: PR-AUC ${fmt(r.pr_auc)} (${ci(r.pr_auc_ci)}) · ` +
        `precision@${a.k} ${fmt(r.precision_at_k)} · base rate ${fmt(r.base_rate, 4)} · ` +
        `${r.n_positive} positives`)));
    return el("tr", {},
      el("td", {}, STAGE_NAMES[s.stage] || s.stage, seasons),
      el("td", { class: "num" }, fmt(m.pr_auc), el("div", { class: "ci" }, ci(m.pr_auc_ci))),
      el("td", { class: "num" }, fmt(m.precision_at_k),
         el("div", { class: "ci" }, ci(m.precision_at_k_ci))),
      el("td", { class: "num" }, m.n_positive.toLocaleString()));
  });
  const table = el("table", {},
    el("thead", {}, el("tr", {}, el("th", {}, "Stage"),
      el("th", { class: "num" }, "PR-AUC (95% CI)"),
      el("th", { class: "num" }, `Precision@${a.k} (95% CI)`),
      el("th", { class: "num" }, "Positive test cell-days"))),
    el("tbody", {}, ...rows));

  const parts = [el("p", { class: "status" },
    `Mean over test seasons ${a.test_seasons.join(", ")}; intervals are ${a.bootstrap.block_days}-day ` +
    `block bootstrap. Top-k is scored per day.`), table];

  const human = a.comparisons.find((c) => c.name === "human_vs_fuel");
  if (human) {
    const d = human.delta_pr_auc, v = human.pr_auc_verdict;
    const text = v === "no detectable difference"
      ? `Adding human-proximity features: no detectable lift (PR-AUC ${d >= 0 ? "+" : ""}${fmt(d)}, ` +
        `95% CI ${ci(human.delta_pr_auc_ci)} spans zero).`
      : `Adding human-proximity features changed PR-AUC by ${d >= 0 ? "+" : ""}${fmt(d)} ` +
        `(95% CI ${ci(human.delta_pr_auc_ci)}).`;
    parts.push(el("p", { class: "headline" }, text));
  }

  const cmpRows = a.comparisons.map((c) => el("tr", {},
    el("td", {}, c.name.replaceAll("_", " ")),
    el("td", {}, `${STAGE_NAMES[c.a] || c.a} vs ${STAGE_NAMES[c.b] || c.b}`),
    el("td", { class: "num" }, `${c.delta_pr_auc >= 0 ? "+" : ""}${fmt(c.delta_pr_auc)}`),
    el("td", { class: "num" }, ci(c.delta_pr_auc_ci)),
    el("td", {}, c.pr_auc_verdict)));
  parts.push(el("h2", {}, "Stage-to-stage differences"), el("table", {},
    el("thead", {}, el("tr", {}, el("th", {}, "Comparison"), el("th", {}, "Stages"),
      el("th", { class: "num" }, "Δ PR-AUC"), el("th", { class: "num" }, "95% CI"),
      el("th", {}, "Verdict"))), el("tbody", {}, ...cmpRows)));

  parts.push(el("h2", {}, "Which family the full model leans on (share of gain)"),
             familyBars(a.importance_by_family));

  parts.push(...spatialSection(a));

  const sens = (a.sensitivity || []).filter((s) => s.status === "ok");
  if (sens.length) {
    parts.push(el("h2", {}, "Robustness to the vegetation threshold"),
      el("p", { class: "status" }, sens.map((s) =>
        `${s.tag.replace("_veg", "")}%: Δ ${fmt(s.delta_pr_auc)} ` +
        `(${ci(s.delta_pr_auc_ci)})`).join("  ·  ")));
  }
  body.replaceChildren(...parts);
}

function spatialSection(a) {
  const sp = a.spatial_holdout || { status: "not_run" };
  const title = el("h2", {}, "Does it work on places it never trained on?");
  if (sp.status !== "ok") {
    return [title, el("p", { class: "status" },
      "Spatial-holdout evaluation has not been run for this build.")];
  }
  const intro = el("p", { class: "status" },
    `Whole ${sp.block_km} km blocks of cells (${sp.n_folds} folds) are held out, so each cell is ` +
    "scored by a model that never saw its neighbourhood, and its own burn history is unavailable.");
  const rows = sp.stages.map((s) => el("tr", {},
    el("td", {}, STAGE_NAMES[s.stage] || s.stage),
    el("td", { class: "num" }, fmt(s.mean.pr_auc), el("div", { class: "ci" }, ci(s.mean.pr_auc_ci))),
    el("td", { class: "num" }, fmt(s.mean.precision_at_k))));
  const table = el("table", {},
    el("thead", {}, el("tr", {}, el("th", {}, "Stage"), el("th", { class: "num" }, "PR-AUC (95% CI)"),
      el("th", { class: "num" }, `Precision@${a.k}`))), el("tbody", {}, ...rows));
  const human = sp.comparisons.find((c) => c.name === "human_vs_fuel");
  const out = [title, intro, table];
  if (human) {
    out.push(el("p", { class: "headline" },
      `On unseen places, adding human-proximity features changes PR-AUC by ` +
      `${human.delta_pr_auc >= 0 ? "+" : ""}${fmt(human.delta_pr_auc)} ` +
      `(95% CI ${ci(human.delta_pr_auc_ci)}).`));
  }
  return out;
}

function familyBars(shares) {
  const names = { history: "Fire history", weather: "Weather", fuel_terrain: "Fuel & terrain",
                  human: "Human proximity" };
  const entries = Object.entries(shares).sort((a, b) => b[1] - a[1]);
  const W = 640, rowH = 30, labelW = 140, barMax = W - labelW - 70;
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${entries.length * rowH + 4}`);
  entries.forEach(([fam, share], i) => {
    const y = i * rowH + 4;
    const label = document.createElementNS(ns, "text");
    label.setAttribute("x", 0); label.setAttribute("y", y + 17);
    label.textContent = names[fam] || fam;
    const rect = document.createElementNS(ns, "rect");
    rect.setAttribute("class", "bar");
    rect.setAttribute("x", labelW); rect.setAttribute("y", y);
    rect.setAttribute("width", Math.max(2, share * barMax)); rect.setAttribute("height", 20);
    rect.setAttribute("rx", 4);
    const value = document.createElementNS(ns, "text");
    value.setAttribute("class", "value");
    value.setAttribute("x", labelW + Math.max(2, share * barMax) + 8); value.setAttribute("y", y + 15);
    value.textContent = `${(share * 100).toFixed(1)}%`;
    rect.addEventListener("mousemove", (e) =>
      showTip(e, names[fam] || fam, [`${(share * 100).toFixed(1)}% of model gain`]));
    rect.addEventListener("mouseout", hideTip);
    svg.append(label, rect, value);
  });
  return el("div", { class: "bars" }, svg);
}

// ---- 3. methodology ---------------------------------------------------------------------
async function initMethodology() {
  const body = $("methodology-body");
  let m;
  try { m = await getJSON("/api/methodology"); } catch (e) { showError(body, e); return; }

  const src = el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Source"),
    el("th", {}, "Window / version"))),
    el("tbody", {}, ...m.data_sources.map((s) => el("tr", {}, el("td", {}, s.name),
      el("td", {}, s.window ? `${s.window.start} to ${s.window.end}` : (s.version || s.source || ""))))));

  const tests = Object.entries(m.leakage_tests).map(([name, t]) => {
    const pass = t.status === "pass";
    const mark = pass ? "✓ pass" : t.status === "fail" ? "✗ FAIL" : "– not run";
    return el("p", {}, el("span", { class: `badge ${pass ? "pass" : "fail"}` }, mark), " ",
      name.replaceAll("_", " "), t.last_run ? ` (last run ${t.last_run})` : "");
  });

  const steps = Object.entries(m.label.filter_steps).map(([k, v]) =>
    el("tr", {}, el("td", {}, k.replaceAll("_", " ")), el("td", { class: "num" }, v.toLocaleString())));

  body.replaceChildren(
    el("h2", {}, "Data sources"), src,
    el("h2", {}, "Grid"),
    el("p", {}, `${m.grid.resolution_km} km cells · ${m.grid.cells_in_state} in the state, ` +
                 `${m.grid.cells_scored} with enough vegetation to score.`),
    el("h2", {}, "Label"), el("p", {}, m.label.definition),
    el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Detection filter step"),
      el("th", { class: "num" }, "Detections left"))), el("tbody", {}, ...steps)),
    el("h2", {}, "Leakage tests"), ...tests,
    el("h2", {}, "Known limitations"),
    el("ul", {}, ...m.known_limitations.map((l) => el("li", {}, l))));
}

route();
