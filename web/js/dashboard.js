"use strict";
(() => {
  const statusEl = document.getElementById("status");
  const f4 = (x) => (typeof x === "number" ? x.toFixed(4) : "-");
  const f3 = (x) => (typeof x === "number" ? x.toFixed(3) : "-");

  function line(parent, text) { const d = document.createElement("div"); d.textContent = text; parent.appendChild(d); }

  function fairness(f) {
    Charts.fairnessBars(document.getElementById("fair"), f);
    const t = document.getElementById("fairtxt"); t.replaceChildren();
    line(t, "Decision rule: " + f.threshold_rule + ".");
    line(t, "Demographic parity difference (DPD): " + f4(f.dpd));
    line(t, "Equalized-odds difference (EOD): " + f4(f.eod) + "  (95% CI " + f4(f.ci95.eod[0]) + " to " + f4(f.ci95.eod[1]) + ")");
    line(t, "Disparate impact ratio (DIR): " + f4(f.dir));
    line(t, "Served seed " + f.served_seed + "; " + f.n_test + " held-out applicants. Seeds differ: see the table below for the 3-seed mean.");
  }

  function calibration(c) {
    Charts.reliability(document.getElementById("calib"), c);
    const t = document.getElementById("caltxt"); t.replaceChildren();
    line(t, "ECE male: " + f4(c.groups.M.ece) + "; ECE female: " + f4(c.groups.F.ece) + "; worst-group ECE: " + f4(c.worst_group_ece) + "; Brier: " + f4(c.brier));
    line(t, "Points on the dashed diagonal are perfectly calibrated.");
  }

  function proxy(p) {
    const t = document.getElementById("proxy"); t.replaceChildren();
    line(t, "Gender predicted from the raw input features: AUC " + f3(p.before_auc_features) + " (0.5 = no signal).");
    line(t, "Gender predicted from the learned representation z (fresh probe, test): A2 AUC " + f3(p.after_auc_z.A2) + "; joint model AUC " + f3(p.after_auc_z.JT) + ".");
    line(t, "Gender predicted from SHAP-attribution vectors: A2 " + f3(p.shap_probe_auc.A2) + "; joint model " + f3(p.shap_probe_auc.JT) + ".");
    line(t, "Top proxy features by gain (gender-from-features probe): " + p.top_proxy_features.slice(0, 6).map((x) => x.feature + " " + (x.gain * 100).toFixed(0) + "%").join(", ") + ".");
    line(t, p.note);
  }

  const CI_KEYS = ["auc_roc", "auc_pr", "worst_group_ece", "eod"];
  const MAIN_ORDER = ["S2-EG", "REF-EG", "A2"];

  function metricCell(r, k) {
    const td = document.createElement("td");
    td.className = "num";
    const lo = r[k + "_lo"], hi = r[k + "_hi"];
    if (typeof lo === "number" && typeof hi === "number") {
      td.textContent = f4(r[k]) + " \u00b1 " + f4((hi - lo) / 2);
      td.title = "95% interval " + f4(lo) + " to " + f4(hi);
    } else td.textContent = f4(r[k]);
    return td;
  }

  function fillTable(tb, rows, served) {
    tb.replaceChildren();
    const tr0 = document.createElement("tr");
    ["Method", "Use", "AUC-ROC", "AUC-PR", "Worst-group ECE", "EOD", "DIR", "Recall"].forEach((h, i) => {
      const th = document.createElement("th"); th.textContent = h; if (i > 1) th.className = "num"; tr0.appendChild(th);
    });
    tb.appendChild(tr0);
    rows.forEach((r) => {
      const tr = document.createElement("tr");
      if (r.arm === served) tr.className = "served";
      const name = document.createElement("td");
      name.textContent = r.arm + (r.arm === served ? " (served)" : "");
      const use = document.createElement("td");
      use.textContent = (r.deployable ? "deployable" : "analysis only") + (r.posthoc ? ", post-hoc" : "");
      use.className = r.deployable ? "dep-yes" : "dep-no";
      tr.append(name, use);
      CI_KEYS.forEach((k) => tr.appendChild(metricCell(r, k)));
      ["dir", "recall"].forEach((k) => tr.appendChild(metricCell(r, k)));
      tb.appendChild(tr);
    });
  }

  function comparison(c, extra) {
    const all = c.rows.concat(extra);
    const main = [c.served_arm, ...MAIN_ORDER].map((a) => all.find((r) => r.arm === a)).filter(Boolean);
    fillTable(document.getElementById("cmp_main"), main, c.served_arm);
    fillTable(document.getElementById("cmp"), all, c.served_arm);
    document.getElementById("honest").textContent = c.headline;
  }

  async function posthocRows() {
    // Separate labelled file: the frozen bundle's comparison stays unchanged. The page still works without it.
    try {
      const res = await fetch("data/posthoc_arms.json");
      return res.ok ? (await res.json()).rows : [];
    } catch (e) { return []; }
  }

  let trade = null, chart = null;
  async function slide() {
    const q = "lambda_f=" + document.getElementById("lf").value + "&lambda_c=" + document.getElementById("lc").value + "&lambda_a=" + document.getElementById("la").value;
    document.getElementById("lfv").textContent = document.getElementById("lf").value;
    document.getElementById("lcv").textContent = document.getElementById("lc").value;
    document.getElementById("lav").textContent = document.getElementById("la").value;
    try {
      const r = await API.get("/tradeoff?" + q);
      const near = trade.points.find((p) => p.lambda_f === r.nearest.lambda_f && p.lambda_c === r.nearest.lambda_c && p.lambda_a === r.nearest.lambda_a);
      if (chart) chart.destroy();
      chart = Charts.tradeoff(document.getElementById("trade"), trade.points, trade.frontier, trade.selected, near);
      const n = document.getElementById("near"); n.replaceChildren();
      line(n, "Nearest sweep point: lambda_f " + r.nearest.lambda_f + ", lambda_c " + r.nearest.lambda_c + ", lambda_a " + r.nearest.lambda_a);
      line(n, "AUC-ROC " + f4(r.metrics.auc_roc) + ", AUC-PR " + f4(r.metrics.auc_pr) + ", EOD " + f4(r.metrics.eod) + ", worst-group ECE " + f4(r.metrics.worst_group_ece) + ", DIR " + f4(r.metrics.dir));
      line(n, "Source: " + r.source + ".");
    } catch (e) { showStatus(statusEl, "err", e.message); }
  }

  async function init() {
    showStatus(statusEl, "info", "Loading dashboard data...");
    try {
      const [f, c, p, cmp, tr, extra] = await Promise.all([API.get("/fairness"), API.get("/calibration"), API.get("/proxy-audit"), API.get("/comparison"), API.get("/tradeoff"), posthocRows()]);
      showStatus(statusEl, "", "");
      trade = tr; fairness(f); calibration(c); proxy(p); comparison(cmp, extra);
      ["lf", "lc", "la"].forEach((id) => document.getElementById(id).addEventListener("input", slide));
      slide();
    } catch (e) {
      showStatus(statusEl, "err", e.code === "API_DOWN" ? "The scoring service is not reachable." : e.message, init);
    }
  }
  init();
})();
