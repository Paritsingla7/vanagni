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
  const name = (location.hash || "#about").slice(1).split("?")[0];
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
const mapState = { map: null, layer: null, fires: null, data: null, ranges: [], req: 0,
                   k: 20, fitted: false };

const DARK = window.matchMedia("(prefers-color-scheme: dark)").matches;
// Sequential ramp, one hue. On a dark surface the low end must recede, so it runs reversed.
const RAMP = DARK ? [...SEQ].reverse() : SEQ;
const OUTLINE = DARK ? "#f0efec" : "#0b0b0b";

// The models' scores are not calibrated probabilities (class weighting inflates them), so
// only the ranking is meaningful. Cells are coloured by their rank within the day.
function rankClass(rank, n, k) {
  if (rank <= k) return 6;
  const p = rank / n;
  return p <= 0.02 ? 5 : p <= 0.05 ? 4 : p <= 0.10 ? 3 : p <= 0.25 ? 2 : p <= 0.50 ? 1 : 0;
}
const classLabels = (k) => ["Lower half", "Upper half", "Top 25%", "Top 10%", "Top 5%", "Top 2%",
                            `Top ${k} cells`];

async function initMap() {
  const status = $("map-status");
  let index;
  try { index = await getJSON("/api/dates"); } catch (e) { showError(status.parentElement, e); return; }
  mapState.ranges = index.ranges;
  mapState.k = index.k || 20;

  mapState.map = L.map("map", { zoomSnap: 0.25 }).setView([30.1, 79.2], 7);
  // A quiet gray basemap (no API key needed) so the data, not the labels, carries the picture;
  // place names go in a pane above the choropleth so they stay readable.
  const shade = DARK ? "Dark" : "Light";
  const esri = (layer) =>
    `https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/${layer}/MapServer/tile/{z}/{y}/{x}`;
  mapState.map.createPane("labels").style.zIndex = 450;
  mapState.map.getPane("labels").style.pointerEvents = "none";
  L.tileLayer(esri(`World_${shade}_Gray_Base`), { maxZoom: 12,
    attribution: "Tiles © Esri — Esri, HERE, Garmin, © OpenStreetMap contributors" })
    .addTo(mapState.map);
  L.tileLayer(esri(`World_${shade}_Gray_Reference`), { maxZoom: 12, pane: "labels" })
    .addTo(mapState.map);

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
  // Shareable deep link, e.g. #map?date=2023-04-12&fires=1
  const want = new URLSearchParams(location.hash.split("?")[1] || "");
  const wanted = mapState.ranges.find((r) => want.get("date") >= r.start && want.get("date") <= r.end);
  seasonSel.value = (wanted || mapState.ranges[mapState.ranges.length - 1]).season;
  setRange();
  if (wanted) dateInp.value = want.get("date");
  $("show-fires").checked = want.get("fires") === "1";
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
  const k = mapState.k, n = data.features.length;
  const ranked = [...data.features].sort((a, b) => b.properties.risk - a.properties.risk);
  const rankOf = new Map(ranked.map((f, i) => [f.properties.cell_id, i + 1]));
  const showFires = $("show-fires").checked;
  if (mapState.layer) mapState.layer.remove();
  if (mapState.fires) mapState.fires.remove();

  const cls = (f) => rankClass(rankOf.get(f.properties.cell_id), n, k);
  mapState.layer = L.geoJSON(data, {
    style: (f) => ({ fillColor: RAMP[cls(f)], fillOpacity: 0.85,
                     weight: cls(f) === 6 ? 2 : 0.3, color: cls(f) === 6 ? OUTLINE : "#898781" }),
    onEachFeature: (f, layer) => {
      layer.on("mousemove", (e) => showTip(e.originalEvent, `Cell ${f.properties.cell_id}`, [
        `Risk rank ${rankOf.get(f.properties.cell_id)} of ${n} today`,
        `Score ${f.properties.risk.toFixed(2)} (not a calibrated probability)`,
        `Fire next day: ${f.properties.fire_next_day ? "yes" : "no"}`]));
      layer.on("mouseout", hideTip);
    },
  }).addTo(mapState.map);
  mapState.layer.eachLayer((l) => { if (cls(l.feature) === 6) l.bringToFront(); });
  if (!mapState.fitted) {                         // zoom to the state once, then leave the view alone
    mapState.map.fitBounds(mapState.layer.getBounds(), { padding: [20, 20], animate: false });
    mapState.fitted = true;
  }

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
  drawLegend(showFires, data, rankOf);
}

function drawLegend(showFires, data, rankOf) {
  const k = mapState.k;
  const burned = data.features.filter((f) => f.properties.fire_next_day);
  const hits = burned.filter((f) => rankOf.get(f.properties.cell_id) <= k).length;
  const labels = classLabels(k);
  const swatches = el("span", { class: "swatches" },
    ...RAMP.map((c, i) => el("span", { style: `background:${c}`, title: labels[i] })));
  const ramp = el("span", { class: "ramp" }, "Lower risk rank", swatches, `Top ${k} cells (outlined)`);
  const parts = [ramp,
    el("span", {}, `${data.features.length} cells · ${burned.length} burned next day · ` +
                    `${hits} of the top ${k} burned`)];
  if (showFires) parts.push(el("span", { class: "ramp" }, el("span", { class: "dot" }),
                                "Actual fire next day"));
  parts.push(el("span", { class: "status" },
    "Colour = rank within the day. Scores are not calibrated probabilities."));
  $("legend").replaceChildren(...parts);
}

// ---- 2. ablation -------------------------------------------------------------------------
const COMPARISON_LABELS = {
  history_vs_persistence: "Does fire history beat the persistence rule?",
  weather_lift: "What does weather add to fire history?",
  fuel_terrain_lift: "What do fuel and terrain add?",
  human_vs_fuel: "What do human-proximity features add? (the hypothesis test)",
  cost_of_weather: "Cost of removing weather from the full model",
  weather_model_vs_fwi: "History + weather model vs the operational FWI",
  full_model_vs_fwi: "Full model vs the operational FWI",
};
const VERDICTS = { "a better": "detectable lift", "b better": "detectable drop",
                   "no detectable difference": "no detectable difference" };

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
    const weather = a.comparisons.find((c) => c.name === "weather_lift");
    const context = weather ? ` For comparison, adding weather changed it by ` +
      `${weather.delta_pr_auc >= 0 ? "+" : ""}${fmt(weather.delta_pr_auc)}.` : "";
    parts.push(el("p", { class: "headline" }, text + context));
  }

  const cmpRows = a.comparisons.map((c) => el("tr", {},
    el("td", {}, COMPARISON_LABELS[c.name] || c.name.replaceAll("_", " ")),
    el("td", { class: "num" }, `${c.delta_pr_auc >= 0 ? "+" : ""}${fmt(c.delta_pr_auc)}`),
    el("td", { class: "num" }, ci(c.delta_pr_auc_ci)),
    el("td", {}, VERDICTS[c.pr_auc_verdict] || c.pr_auc_verdict)));
  parts.push(el("h2", {}, "Stage-to-stage differences (change in PR-AUC)"), el("p", { class: "status" },
    "Intervals capture day-to-day sampling noise within each test season. They do not capture " +
    "season-to-season variation, which matters with only a few seasons: treat small lifts with caution."),
    el("table", {},
    el("thead", {}, el("tr", {}, el("th", {}, "Question"),
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
    const kind = t.status === "pass" ? "pass" : t.status === "fail" ? "fail" : "notrun";
    const mark = { pass: "✓ pass", fail: "✗ FAIL", notrun: "– not run" }[kind];
    return el("p", {}, el("span", { class: `badge ${kind}` }, mark), " ",
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
