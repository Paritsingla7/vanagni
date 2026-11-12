"use strict";

// The briefing page: fills its numbers from the same result files the site reads (so they
// follow every re-run), and draws the stage ladders and the rolling-origin strip. The text
// written into the page is the fallback if a file can't be read.

const STATIC = document.documentElement.dataset.static === "1";
const getJSON = async (url) => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return res.json();
};
// Static site: /data/<region>/...; Flask: /api/...?region=<region>.
const fileFor = (rid, name) => STATIC
  ? `/data/${rid}/${{ ablation: "ablation.json", methodology: "methodology.json",
                      dates: "risk/index.json" }[name]}`
  : `/api/${name}?region=${rid}`;

const f3 = (x) => Number(x).toFixed(3).replace("-", "−");
const signed = (x) => `${x >= 0 ? "+" : ""}${f3(x)}`;
const nf = (x) => Number(x).toLocaleString("en-GB");
const ciText = (ci) => `CI ${f3(ci[0])} – ${f3(ci[1])}`;
const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight"];

function put(key, text) {
  document.querySelectorAll(`[data-v="${key}"]`).forEach((n) => { n.textContent = text; });
}

const STAGES = [["1", "Persistence"], ["2", "+ fire history"], ["3", "+ weather"],
                ["4a", "+ fuel & terrain"], ["4b", "+ human proximity"]];

function ladder(rid, byStage) {
  const box = document.querySelector(`[data-ladder="${rid}"]`);
  if (!box) return;
  const max = Math.max(...STAGES.map(([k]) => byStage[k] || 0));
  let prev = null;
  box.replaceChildren(...STAGES.filter(([k]) => byStage[k] != null).map(([k, name]) => {
    const v = byStage[k], step = prev == null ? null : v - prev;
    prev = v;
    const row = document.createElement("div");
    row.className = `rung${k === "3" ? " key" : ""}${k === "4b" ? " test" : ""}`;
    const label = document.createElement("span");
    label.className = "name"; label.textContent = name;
    const bar = document.createElement("span");
    bar.className = "bar"; bar.style.setProperty("--w", `${(v / max) * 100}%`);
    const val = document.createElement("span");
    val.className = "val";
    val.textContent = f3(v) + (step == null ? "" : `  (${signed(step)})`);
    row.append(label, bar, val);
    return row;
  }));
}

function rolling(ranges, firstYear) {
  const box = document.querySelector("[data-roll]");
  if (!box || !ranges.length) return;
  const last = ranges.at(-1).season;
  const years = Array.from({ length: last - firstYear + 1 }, (_, i) => firstYear + i);
  const rows = ranges.map((r) => {
    const row = document.createElement("div");
    row.className = "roll-row";
    for (const y of years) {
      const c = document.createElement("span");
      c.className = y < r.season ? "tr" : y === r.season ? "te" : "no";
      c.title = `${y}`;
      row.append(c);
    }
    return row;
  });
  const key = document.createElement("p");
  key.className = "roll-key";
  key.innerHTML = '<span class="tr"></span> train <span class="te"></span> test ' +
                  `<span class="no"></span> unused &nbsp; ${firstYear}–${last}, one row per test season`;
  box.replaceChildren(...rows, key);
}

async function region(rid) {
  const [a, m, d] = await Promise.all(["ablation", "methodology", "dates"]
    .map((name) => getJSON(fileFor(rid, name))));
  const cmp = (list, name) => (list || []).find((c) => c.name === name);
  const stage = Object.fromEntries(a.stages.map((s) => [s.stage, s.mean]));
  const w = cmp(a.comparisons, "weather_lift"), h = cmp(a.comparisons, "human_vs_fuel");
  const seasons = d.ranges.map((r) => r.season);
  const n = WORDS[seasons.length] || String(seasons.length);
  put(`${rid}.span`, `${n} seasons, ${seasons[0]}–${seasons.at(-1)}`);
  put(`${rid}.weather`, signed(w.delta_pr_auc));
  put(`${rid}.weather_ci`, ciText(w.delta_pr_auc_ci));
  put(`${rid}.human`, signed(h.delta_pr_auc));
  put(`${rid}.human_ci`, ciText(h.delta_pr_auc_ci));
  if (h.delta_pr_auc > 0) put(`${rid}.ratio`, `${Math.round(w.delta_pr_auc / h.delta_pr_auc)}×`);
  const sp = a.spatial_holdout?.status === "ok" ? cmp(a.spatial_holdout.comparisons, "human_vs_fuel") : null;
  if (sp) put(`${rid}.human_spatial`, signed(sp.delta_pr_auc));
  put(`${rid}.full`, f3(stage["4b"].pr_auc));
  put(`${rid}.fwi`, f3(stage.fwi.pr_auc));
  put(`${rid}.persist`, f3(stage["1"].pr_auc));
  put(`${rid}.vs_fwi`, `${Math.round(stage["4b"].pr_auc / stage.fwi.pr_auc)}×`);
  put(`${rid}.p20`, (stage["4b"].precision_at_k * a.k).toFixed(1));
  put(`${rid}.pos`, nf(stage["4b"].n_positive));
  const per = Object.values(h.delta_pr_auc_per_season);
  put(`${rid}.positive_seasons`,
      `${per.filter((x) => Number(x.toFixed(3)) > 0).length} of ${per.length}`);
  const share = a.importance_by_family || {};
  if (share.human != null) put(`${rid}.share_human`, `${(share.human * 100).toFixed(1)}%`);
  if (share.history != null) put(`${rid}.share_history`, `${(share.history * 100).toFixed(1)}%`);
  if (m.grid?.cells_scored) put(`${rid}.cells`, nf(m.grid.cells_scored));
  const steps = Object.values(m.label?.filter_steps || {});
  if (steps.length) { put(`${rid}.f_raw`, nf(steps[0])); put(`${rid}.f_kept`, nf(steps.at(-1))); }
  ladder(rid, Object.fromEntries(Object.entries(stage).map(([k, v]) => [k, v.pr_auc])));
  return { ranges: d.ranges, firstYear: Number((m.data_sources?.[0]?.window?.start || "2017").slice(0, 4)) };
}

// Gentle entrances, as on the main site.
function reveal() {
  document.documentElement.classList.add("js");
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
  }), { threshold: 0.12 });
  document.querySelectorAll(".bf-sec .bf-body, .bf-end").forEach((t) => io.observe(t));
}

(async () => {
  reveal();
  const results = await Promise.allSettled(["uttarakhand", "odisha"].map(region));
  const uk = results[0].status === "fulfilled" ? results[0].value : null;
  if (uk) rolling(uk.ranges, uk.firstYear);
})();
