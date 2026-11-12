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

async function fetchJSON(url) {
  const res = await fetch(url);
  const body = await res.json();
  if (!res.ok) throw new Error(body.message || `${res.status}`);
  return body;
}

// Static hosting (the exported site, e.g. on Vercel): no server, so the same pre-computed
// files are read directly. `wfpin.serve.export` sets data-static on <html>; the Flask app
// never does, so it keeps calling its /api endpoints unchanged.
const STATIC = document.documentElement.dataset.static === "1";
const staticCache = {};
const cached = (url) => (staticCache[url] ||= fetchJSON(url).catch((e) => {
  delete staticCache[url]; throw e;
}));

async function staticRisk(base, day) {
  const index = await cached(`${base}/risk/index.json`);
  if (!index.ranges.some((r) => r.start <= day && day <= r.end)) {
    const valid = index.ranges.map((r) => `${r.start} to ${r.end}`).join(", ");
    throw new Error(`No prediction exists for ${day}. Valid dates: ${valid}.`);
  }
  const [grid, payload] = await Promise.all([cached(`${base}/risk/grid.geojson`),
                                             fetchJSON(`${base}/risk/${day}.json`)]);
  const geometry = grid._byId ||= Object.fromEntries(
    grid.features.map((f) => [String(f.properties.cell_id), f.geometry]));
  return { type: "FeatureCollection", date: payload.date,
           features: Object.entries(payload.cells).map(([cid, [r, y]]) => ({
             type: "Feature", geometry: geometry[cid],
             properties: { cell_id: Number(cid), risk: r, fire_next_day: y } })) };
}

// ---- regions: the landing tells one region's story (the default); the map, results and
// method pages follow the region switch. `id` null = a server without regions (one region).
const region = { list: [], default: null, current: null };
const regionName = (id) => (region.list.find((r) => r.id === id) || {}).name || "";

async function loadRegions() {
  try {
    const r = await (STATIC ? cached("/data/regions.json") : fetchJSON("/api/regions"));
    region.list = r.regions;
    region.default = r.default;
  } catch (e) { return; }                       // the pages still work for a single region
  const want = new URLSearchParams(location.hash.split("?")[1] || "").get("region");
  region.current = region.list.some((r) => r.id === want) ? want : region.default;
}

// `rid` defaults to the switched-to region; the landing passes region.default.
async function getJSON(url, rid = region.current) {
  const [path, query] = url.split("?");
  const params = new URLSearchParams(query);
  if (!STATIC) {
    if (rid) params.set("region", rid);
    return fetchJSON(params.size ? `${path}?${params}` : path);
  }
  const base = rid ? `/data/${rid}` : "/data";
  if (path === "/api/ablation") return cached(`${base}/ablation.json`);
  if (path === "/api/methodology") return cached(`${base}/methodology.json`);
  if (path === "/api/dates") return cached(`${base}/risk/index.json`);
  if (path === "/api/risk") return staticRisk(base, params.get("date") || "");
  throw new Error(`No static file for ${path}`);
}

// A season that wraps the new year (Odisha's Nov-Jun) reads "2020–21"; one inside a year, "2024".
const seasonLabel = (r) => {
  const a = r.start.slice(0, 4), b = r.end.slice(0, 4);
  return a === b ? String(r.season) : `${a}–${b.slice(2)}`;
};

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
  document.body.classList.toggle("on-landing", key === "about");
  document.querySelectorAll(".topbar .links a").forEach((a) =>
    a.classList.toggle("active", a.getAttribute("href") === `#${key}`));
  $("region-switch").hidden = key === "about" || region.list.length < 2;
  if (!loaded[key]) {
    loaded[key] = true;
    ({ about: initAbout, map: initMap, ablation: initAblation,
       methodology: initMethodology }[key] || (() => {}))();
  } else if (key === "map" && mapState.map) {
    mapState.map.invalidateSize();
  }
}
window.addEventListener("hashchange", route);

// The switch: a region change re-loads whichever app screen is open, and the others on visit.
function initRegionSwitch() {
  const box = $("region-switch");
  box.replaceChildren(...region.list.map((r) => {
    const b = el("button", { type: "button", "data-region": r.id }, r.name);
    b.addEventListener("click", () => setRegion(r.id));
    return b;
  }));
  markRegion();
}

function markRegion() {
  $("region-switch").querySelectorAll("button").forEach((b) =>
    b.setAttribute("aria-pressed", String(b.dataset.region === region.current)));
  fill("region", regionName(region.current));
  const note = (region.list.find((r) => r.id === region.current) || {}).note || "";
  document.querySelectorAll(".region-note").forEach((n) => { n.textContent = note; n.hidden = !note; });
}

function setRegion(id) {
  if (id === region.current) return;
  region.current = id;
  markRegion();
  for (const key of ["map", "ablation", "methodology"]) loaded[key] = false;
  const [screen] = (location.hash || "#about").slice(1).split("?");
  // Keep the region in the link (shareable); a map date from the other region would not apply.
  history.replaceState(null, "", `#${screen}?region=${id}`);
  route();
}

// ---- 0. about: result at a glance (computed from the real results, never typed in) ------
// Every region the site serves is shown side by side: the finding holding in more than one
// state is the point, so the landing page leads with all of them, not just the default.

// A lift below 0.001 would print as "+0.000"; show it to four places instead.
const fmtLift = (x) => fmt(x, x !== 0 && Math.abs(x) < 0.001 ? 4 : 3);
const signed = (x) => `${x >= 0 ? "+" : ""}${fmtLift(x)}`;
// Seasons with a lift at the precision shown (3 dp): +0.0003 is not a lift anyone could see.
const positiveSeasons = (xs) => xs.filter((x) => Number(x.toFixed(3)) > 0).length;
const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
               "eleven", "twelve"];
const word = (n) => WORDS[n] || String(n);
const cap = (t) => `${t[0].toUpperCase()}${t.slice(1)}`;
const andList = (xs) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs.at(-1)}`);
const nf = (x) => Number(x).toLocaleString("en-GB");
const fill = (key, text) => document.querySelectorAll(`[data-fill="${key}"]`)
  .forEach((n) => { n.textContent = text; });

// Each region's results, fetched once and settled, so one missing region hides only itself.
async function allRegions(fetchOne) {
  const out = await Promise.allSettled(region.list.map((r) => fetchOne(r.id)));
  return region.list.map((r, i) => ({ ...r, ...(out[i].status === "fulfilled" ? out[i].value : {}),
                                      ok: out[i].status === "fulfilled" }));
}

// One sentence per region: what human proximity did, read from its interval.
function regionVerdict(r) {
  const [lo] = r.human.delta_pr_auc_ci, h = r.human.delta_pr_auc, w = r.weather.delta_pr_auc;
  if (h > w && lo > 0) return { yes: true, text: `in ${r.name} human proximity added ${(h / w).toFixed(1)} times the lift of weather` };
  if (lo <= 0) return { yes: false, text: `in ${r.name} their lift could not be told apart from zero` };
  return { yes: false, text: `in ${r.name} weather added about ${Math.round(w / h)} times more than roads, villages and fields did` };
}

async function initAbout() {
  fillCopy();
  initMoments();
  initReveal();
  const regions = (await allRegions(async (rid) => {
    const [a, d] = await Promise.all([getJSON("/api/ablation", rid), getJSON("/api/dates", rid)]);
    const cmp = (name) => a.comparisons.find((c) => c.name === name);
    return { a, seasons: d.ranges.map((r) => r.season), human: cmp("human_vs_fuel"),
             weather: cmp("weather_lift"), fwi: cmp("full_model_vs_fwi") };
  })).filter((r) => r.ok);
  if (!regions.length) {
    $("facts").replaceChildren(el("p", { class: "status" }, "Results have not been built for this run yet."));
    initBulletin();
    return;
  }
  const fact = (n, label) => el("div", { class: "fact" },
    el("span", { class: "fact-n" }, n), el("span", { class: "fact-l" }, label));
  $("facts").replaceChildren(...regions.map((r) => el("div", { class: "fact-state reveal" },
    el("h3", {}, r.name, el("small", {},
      `${word(r.seasons.length)} test seasons, ${r.seasons[0]}–${r.seasons.at(-1)}`)),
    el("div", { class: "fact-row" },
      fact(signed(r.weather.delta_pr_auc), `PR-AUC added by weather (95% CI ${ci(r.weather.delta_pr_auc_ci)})`),
      fact(signed(r.human.delta_pr_auc), `added by roads, villages and fields (95% CI ${ci(r.human.delta_pr_auc_ci)})`),
      fact(signed(r.fwi.delta_pr_auc), "the full model over the operational Fire Weather Index")))));
  const verdicts = regions.map(regionVerdict);
  const all = regions.length > 1 ? (regions.length === 2 ? "in both states" : "in every state") : "";
  const head = verdicts.every((v) => !v.yes) ? `The answer is no${all ? `, ${all}` : ""}.`
    : verdicts.every((v) => v.yes) ? `The answer is yes${all ? `, ${all}` : ""}.`
      : "The answer differs by state.";
  $("verdict").textContent = `${head} ${cap(verdicts.map((v) => v.text).join("; "))}. ` +
    "Fire history and the weather carry tomorrow.";

  // The ledger: each stage's PR-AUC, one column per state.
  const byStage = regions.map((r) => Object.fromEntries(r.a.stages.map((s) => [s.stage, s.mean.pr_auc])));
  const rows = [
    ["fwi", "The Fire Weather Index as a rule", "the operational dryness rating", "dim"],
    ["1", "Burning today, burning tomorrow", "the persistence rule", "dim"],
    ["2", "Add fire history", "recent fires in and around the cell", ""],
    ["3", "Add weather", "heat, humidity, wind, rain, soil moisture", "key"],
    ["4a", "Add fuel and terrain", "forest cover, slope, elevation", ""],
    ["4b", "Add human proximity", "roads, settlements, cropland", ""],
  ];
  const ledger = $("ledger");
  ledger.style.setProperty("--n", regions.length);
  ledger.replaceChildren(
    el("div", { class: "lrow lhead" }, el("span", { class: "cap" }),
      ...regions.map((r) => el("span", { class: "v" }, r.name))),
    ...rows.filter(([k]) => byStage.some((b) => b[k] != null)).map(([k, title, note, cls], i) =>
      el("div", { class: `lrow reveal ${cls}`, style: `--d:${i * 70}ms` },
        el("span", { class: "cap" }, title, el("small", {}, note)),
        ...byStage.map((b) => el("span", { class: "v" }, b[k] == null ? "–" : fmt(b[k]))))));
  initReveal();
  initBulletin();
}

// ---- counts in the page copy, summed over every region, from the same files -------------
async function fillCopy() {
  const regions = (await allRegions(async (rid) => {
    const [index, m] = await Promise.all([getJSON("/api/dates", rid), getJSON("/api/methodology", rid)]);
    return { index, m };
  })).filter((r) => r.ok);
  if (!regions.length) return;   // the written defaults stay
  const names = regions.map((r) => r.name);
  fill("states", andList(names));
  fill("states-n", word(regions.length));
  const counts = regions.map((r) => r.index.ranges.length);
  fill("seasons-cap", cap(regions.map((r, i) =>
    `${word(counts[i])}${i === 0 ? " fire" : ""} seasons in ${r.name}`).reduce((t, x, i, xs) =>
    (i === 0 ? x : `${t}${i === xs.length - 1 ? " and " : ", "}${x}`), "")));
  fill("seasons-total", word(counts.reduce((a, b) => a + b, 0)));
  fill("seasons-n", word(counts[0]));
  const sum = (f) => regions.reduce((t, r) => t + (f(r.m) || 0), 0);
  const first = (m) => Object.values(m.label?.filter_steps || {})[0];
  const last = (m) => Object.values(m.label?.filter_steps || {}).at(-1);
  if (sum(first)) { fill("raw", nf(sum(first))); fill("kept", nf(sum(last))); }
  if (sum((m) => m.grid?.cells_scored)) fill("cells", nf(sum((m) => m.grid?.cells_scored)));
  const windows = regions.map((r) => (r.m.data_sources || []).find((d) => /FIRMS/.test(d.name))?.window)
    .filter(Boolean);
  if (windows.length) {
    const years = windows.flatMap((w) => [w.start, w.end].map((x) => Number(x.slice(0, 4))));
    fill("window", `${Math.min(...years)} to ${Math.max(...years)}`);
  }
}

// ---- the live inspection sheet: the real top 20 for any test day, in any state ----------
// The day each state's sheet opens on: a peak-season day; any test day works.
const SHEET_DAY = { uttarakhand: "2024-04-20", odisha: "2024-03-29" };
const bulletin = { ranges: [], req: 0, region: null };
const centroid = (geom) => {
  const ring = geom.coordinates[0];
  return [ring.reduce((s, p) => s + p[1], 0) / ring.length, ring.reduce((s, p) => s + p[0], 0) / ring.length];
};
const longDate = (iso) => new Date(`${iso}T00:00:00`).toLocaleDateString("en-GB",
  { day: "numeric", month: "long", year: "numeric" });
const shiftDay = (iso, n) => {
  const d = new Date(`${iso}T00:00:00Z`); d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
};
const rangeDays = (ranges) => ranges.reduce((s, r) =>
  s + Math.round((Date.parse(r.end) - Date.parse(r.start)) / 864e5) + 1, 0);
const sheetValid = (iso) => bulletin.ranges.some((r) => iso >= r.start && iso <= r.end);

async function initBulletin() {
  const inp = $("b-date");
  if (inp.dataset.ready) return;
  inp.dataset.ready = "1";
  const box = $("sheet-switch");
  if (region.list.length > 1) {
    box.replaceChildren(...region.list.map((r) => {
      const b = el("button", { type: "button", "data-region": r.id }, r.name);
      b.addEventListener("click", () => setSheetRegion(r.id));
      return b;
    }));
    box.hidden = false;
  }
  allRegions((rid) => getJSON("/api/dates", rid)).then((rs) => {
    const days = rs.filter((r) => r.ok).reduce((t, r) => t + rangeDays(r.ranges), 0);
    if (days) $("n-days").textContent = nf(days);
  });
  const step = (n) => {
    let d = shiftDay(inp.value, n);
    // jump the gap between seasons
    for (let i = 0; i < 400 && !sheetValid(d) && d >= inp.min && d <= inp.max; i++) d = shiftDay(d, n);
    if (sheetValid(d)) { inp.value = d; loadSheet(); }
  };
  $("b-prev").addEventListener("click", () => step(-1));
  $("b-next").addEventListener("click", () => step(1));
  inp.addEventListener("change", () => { if (sheetValid(inp.value)) loadSheet(); });
  setSheetRegion(region.default);
}

async function setSheetRegion(rid) {
  if (rid === bulletin.region) return;
  bulletin.region = rid;
  $("sheet-switch").querySelectorAll("button").forEach((b) =>
    b.setAttribute("aria-pressed", String(b.dataset.region === rid)));
  $("sheet-region").textContent = regionName(rid);
  let index;
  try { index = await getJSON("/api/dates", rid); } catch (e) {
    $("sheet-date").textContent = "The risk files have not been built yet.";
    return;
  }
  if (rid !== bulletin.region) return;
  bulletin.ranges = index.ranges;
  const inp = $("b-date");
  inp.min = index.ranges[0].start;
  inp.max = index.ranges.at(-1).end;
  const last = index.ranges.at(-1);
  const want = SHEET_DAY[rid];
  inp.value = want && sheetValid(want) ? want : shiftDay(last.start, Math.round(rangeDays([last]) * 0.6));
  loadSheet();
}

async function loadSheet() {
  const iso = $("b-date").value, ticket = ++bulletin.req;
  $("sheet-date").textContent = `for ${longDate(shiftDay(iso, 1))}`;
  let data;
  try { data = await getJSON(`/api/risk?date=${iso}`, bulletin.region); } catch (e) {
    if (ticket === bulletin.req) $("sheet-foot").textContent = e.message;
    return;
  }
  if (ticket !== bulletin.req) return;
  const ranked = [...data.features].sort((x, y) => y.properties.risk - x.properties.risk);
  const top = ranked.slice(0, 20);
  const hits = top.filter((f) => f.properties.fire_next_day).length;
  const burned = data.features.filter((f) => f.properties.fire_next_day).length;
  $("sheet-rows").replaceChildren(...top.map((f, i) => {
    const [lat, lon] = centroid(f.geometry), fire = f.properties.fire_next_day;
    return el("li", { class: fire ? "burned" : "" },
      el("span", { class: "no" }, String(i + 1)),
      el("span", {}, `Cell ${f.properties.cell_id} · ${lat.toFixed(2)}° N, ${lon.toFixed(2)}° E`),
      el("span", { class: fire ? "hit" : "miss" }, fire ? "burned" : "no fire"));
  }));
  const share = (burned / data.features.length * 100).toFixed(1);
  $("sheet-foot").textContent = `${hits} of these 20 burned. Across all ${data.features.length.toLocaleString("en-GB")} ` +
    `cells, ${burned} did (${share}%), so a random list of 20 would expect about ${(burned / data.features.length * 20).toFixed(1)}.`;
}

// ---- the drag row and the entrances ---------------------------------------------------
function initMoments() {
  const row = $("moments"), bar = $("m-bar");
  if (!row || row.dataset.ready) return;
  row.dataset.ready = "1";
  const update = () => {
    const max = row.scrollWidth - row.clientWidth;
    const seen = (row.clientWidth + row.scrollLeft) / row.scrollWidth;
    bar.style.width = `${Math.min(100, (max <= 0 ? 1 : seen) * 100)}%`;
  };
  const by = () => (row.querySelector("li")?.offsetWidth || 300) + 32;
  $("m-back").addEventListener("click", () => row.scrollBy({ left: -by(), behavior: "smooth" }));
  $("m-fwd").addEventListener("click", () => row.scrollBy({ left: by(), behavior: "smooth" }));
  row.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);
  update();
}

function initReveal() {
  document.documentElement.classList.add("js");
  const targets = document.querySelectorAll(
    ".landing h2, .intro-note, .verdict, .intro-photo, .bulletin-copy, .sheet, .ledger-note," +
    " .moments-row li, .card-list, .reply-note, .reply-card, .back-card, .reveal");
  const io = new IntersectionObserver((entries) => entries.forEach((e) => {
    if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
  }), { threshold: 0.15 });
  targets.forEach((t, i) => {
    if (t.classList.contains("in")) return;
    t.classList.add("reveal");
    if (!t.style.getPropertyValue("--d")) t.style.setProperty("--d", `${(i % 4) * 80}ms`);
    io.observe(t);
  });
}

// ---- 1. risk map -----------------------------------------------------------------------
const mapState = { map: null, layer: null, fires: null, data: null, ranges: [], req: 0,
                   k: 20, fitted: false };

// The page is a light, printed object in every colour scheme, so the map follows it.
const DARK = false;
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

// Built once; a region switch only reloads the dates and the cells (setupMap is idempotent).
function setupMap() {
  if (mapState.map) return;
  mapState.map = L.map("map-canvas", { zoomSnap: 0.25 }).setView([30.1, 79.2], 7);
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
  seasonSel.addEventListener("change", () => { setRange(); loadDate(); });
  dateInp.addEventListener("change", loadDate);
  $("show-fires").addEventListener("change", draw);
}

function setRange() {
  const seasonSel = $("season"), dateInp = $("date");
  const r = mapState.ranges.find((x) => String(x.season) === seasonSel.value);
  dateInp.min = r.start; dateInp.max = r.end;   // only dates with a prediction are selectable
  if (!dateInp.value || dateInp.value < r.start || dateInp.value > r.end) {
    dateInp.value = r.start;
  }
}

async function initMap() {
  const status = $("map-status");
  let index;
  try { index = await getJSON("/api/dates"); } catch (e) { showError(status.parentElement, e); return; }
  mapState.ranges = index.ranges;
  mapState.k = index.k || 20;
  setupMap();
  mapState.map.invalidateSize();
  // A new region: drop the last one's cells and zoom to the new state on the first draw.
  mapState.layer?.remove(); mapState.fires?.remove();
  mapState.layer = mapState.fires = mapState.data = null;
  mapState.fitted = false;

  const seasonSel = $("season"), dateInp = $("date");
  seasonSel.replaceChildren(...mapState.ranges.map((r) =>
    el("option", { value: r.season }, seasonLabel(r))));
  // Shareable deep link, e.g. #map?region=odisha&date=2023-04-12&fires=1
  const want = new URLSearchParams(location.hash.split("?")[1] || "");
  const wanted = mapState.ranges.find((r) => want.get("date") >= r.start && want.get("date") <= r.end);
  seasonSel.value = (wanted || mapState.ranges[mapState.ranges.length - 1]).season;
  dateInp.value = wanted ? want.get("date") : "";
  setRange();
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
  fwi_feature_lift: "What does the standard FWI add to fire history?",
  raw_weather_vs_fwi_feature: "Do the raw weather variables beat the FWI inside the same model?",
  full_model_vs_fwi: "Full model vs the operational FWI",
};
const VERDICTS = { "a better": "detectable lift", "b better": "detectable drop",
                   "no detectable difference": "no detectable difference" };

const STAGE_NAMES = { "1": "Persistence", "fwi": "Fire Weather Index (operational baseline)",
                      "2": "+ Fire history", "3": "+ Weather",
                      "3f": "History + FWI (one index instead of raw weather)",
                      "4a": "+ Fuel & terrain", "4b": "+ Human proximity",
                      "4b-weather": "Full model minus weather" };

async function initAblation() {
  const body = $("ablation-body");
  let a, labels = {};
  try { a = await getJSON("/api/ablation"); } catch (e) { showError(body, e); return; }
  try {
    const index = await getJSON("/api/dates");
    labels = Object.fromEntries(index.ranges.map((r) => [r.season, seasonLabel(r)]));
  } catch (e) { /* plain season numbers */ }
  const label = (season) => labels[season] || String(season);

  const rows = a.stages.map((s) => {
    const m = s.mean;
    const seasons = el("details", {}, el("summary", {}, "per season"),
      ...s.per_season.map((r) => el("div", { class: "ci" },
        `${label(r.season)}: PR-AUC ${fmt(r.pr_auc)} (${ci(r.pr_auc_ci)}) · ` +
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
    `Mean over test seasons ${a.test_seasons.map(label).join(", ")}; intervals are ${a.bootstrap.block_days}-day ` +
    `block bootstrap. Top-k is scored per day.`), table];

  const human = a.comparisons.find((c) => c.name === "human_vs_fuel");
  if (human) {
    const d = human.delta_pr_auc, v = human.pr_auc_verdict;
    const text = v === "no detectable difference"
      ? `Adding human-proximity features: no detectable lift (PR-AUC ${signed(d)}, ` +
        `95% CI ${ci(human.delta_pr_auc_ci)} spans zero).`
      : `Adding human-proximity features changed PR-AUC by ${signed(d)} ` +
        `(95% CI ${ci(human.delta_pr_auc_ci)}).`;
    const weather = a.comparisons.find((c) => c.name === "weather_lift");
    const perSeason = Object.values(human.delta_pr_auc_per_season);
    const positive = positiveSeasons(perSeason);
    const context = (weather ? ` For comparison, adding weather changed it by ` +
      `${signed(weather.delta_pr_auc)}.` : "") +
      ` The human-proximity change was positive in ${positive} of ${perSeason.length} test seasons.`;
    parts.push(el("p", { class: "headline" }, text + context));
  }

  const cmpRows = a.comparisons.map((c) => el("tr", {},
    el("td", {}, COMPARISON_LABELS[c.name] || c.name.replaceAll("_", " ")),
    el("td", { class: "num" }, `${signed(c.delta_pr_auc)}`),
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
      `${signed(human.delta_pr_auc)} ` +
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
function gapText(source) {
  if (!source.gaps || !source.gaps.length) return "";
  return " (not downloaded: " + source.gaps.map((g) => `${g.start} to ${g.end}`).join("; ") +
         " — outside every fire season)";
}

async function initMethodology() {
  const body = $("methodology-body");
  let m;
  try { m = await getJSON("/api/methodology"); } catch (e) { showError(body, e); return; }

  const src = el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Source"),
    el("th", {}, "Window / version"))),
    el("tbody", {}, ...m.data_sources.map((s) => el("tr", {}, el("td", {}, s.name),
      el("td", {}, s.window ? `${s.window.start} to ${s.window.end}` + gapText(s) :
                 (s.version || s.source || ""))))));

  const tests = Object.entries(m.leakage_tests).map(([name, t]) => {
    const kind = t.status === "pass" ? "pass" : t.status === "fail" ? "fail" : "notrun";
    const mark = { pass: "Passed", fail: "FAILED", notrun: "Not run" }[kind];
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

loadRegions().then(() => { initRegionSwitch(); route(); });
