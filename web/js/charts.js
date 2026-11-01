"use strict";
// Chart helpers (Chart.js 4.4.1 from a pinned CDN URL). Charts are drawn only when Chart.js loaded.
const Charts = (() => {
  const colors = { pos: "#b3261e", neg: "#1f5fbf", m: "#1f5fbf", f: "#a8326f", pt: "#9aa5b5", sel: "#1f7a4d" };
  function ready() { return typeof Chart !== "undefined"; }
  function placeholder(canvas, msg) {
    const p = document.createElement("p");
    p.className = "muted small"; p.textContent = msg;
    canvas.replaceWith(p);
  }
  function shapBars(canvas, items) {
    if (!ready()) return placeholder(canvas, "Chart library unavailable (offline): " + items.map((i) => i.feature + " " + i.shap.toFixed(3)).join("; "));
    const top = items.slice(0, 10);
    return new Chart(canvas, {
      type: "bar",
      data: { labels: top.map((i) => i.feature + (i.value === null ? "" : " = " + i.value)),
              datasets: [{ data: top.map((i) => i.shap), backgroundColor: top.map((i) => (i.shap >= 0 ? colors.pos : colors.neg)) }] },
      options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false } },
                 scales: { x: { title: { display: true, text: "SHAP contribution to default logit" } }, y: { ticks: { font: { size: 11 } } } } },
    });
  }
  function fairnessBars(canvas, fair) {
    if (!ready()) return placeholder(canvas, "Chart library unavailable (offline).");
    const g = fair.per_group;
    return new Chart(canvas, {
      type: "bar",
      data: { labels: ["TPR", "FPR"], datasets: [
        { label: "Male", data: [g.M.tpr, g.M.fpr], backgroundColor: colors.m },
        { label: "Female", data: [g.F.tpr, g.F.fpr], backgroundColor: colors.f }] },
      options: { maintainAspectRatio: false, scales: { y: { beginAtZero: true, title: { display: true, text: "rate at the flag threshold" } } } },
    });
  }
  function reliability(canvas, cal) {
    if (!ready()) return placeholder(canvas, "Chart library unavailable (offline).");
    const ds = (name, col, g) => ({ label: name + " (ECE " + g.ece.toFixed(4) + ")", data: g.bins.map((b) => ({ x: b.mean_pred, y: b.frac_pos })), borderColor: col, backgroundColor: col, showLine: true, tension: 0.15 });
    const mx = Math.max(...cal.groups.M.bins.concat(cal.groups.F.bins).map((b) => Math.max(b.mean_pred, b.frac_pos)), 0.1) * 1.1;
    return new Chart(canvas, {
      type: "scatter",
      data: { datasets: [ds("Male", colors.m, cal.groups.M), ds("Female", colors.f, cal.groups.F),
        { label: "Perfect", data: [{ x: 0, y: 0 }, { x: mx, y: mx }], borderColor: colors.pt, borderDash: [5, 5], pointRadius: 0, showLine: true }] },
      options: { maintainAspectRatio: false, scales: { x: { min: 0, max: mx, title: { display: true, text: "mean predicted probability" } }, y: { min: 0, max: mx, title: { display: true, text: "observed default rate" } } } },
    });
  }
  function tradeoff(canvas, points, frontier, selected, nearest) {
    if (!ready()) return placeholder(canvas, "Chart library unavailable (offline).");
    const xy = (p) => ({ x: p.eod, y: p.auc_pr });
    return new Chart(canvas, {
      type: "scatter",
      data: { datasets: [
        { label: "sweep points", data: points.map(xy), backgroundColor: colors.pt, pointRadius: 4 },
        { label: "non-dominated", data: frontier.map(xy), backgroundColor: colors.m, pointRadius: 5 },
        { label: "deployed (0.05, 2, 0)", data: [xy(selected)], backgroundColor: colors.sel, pointRadius: 9, pointStyle: "rectRot" },
        { label: "nearest to sliders", data: nearest ? [xy(nearest)] : [], backgroundColor: "#d97706", pointRadius: 8, pointStyle: "triangle" }] },
      options: { maintainAspectRatio: false, animation: false, scales: { x: { title: { display: true, text: "EOD (lower is fairer)" } }, y: { title: { display: true, text: "AUC-PR (higher is better)" } } } },
    });
  }
  return { shapBars, fairnessBars, reliability, tradeoff };
})();
