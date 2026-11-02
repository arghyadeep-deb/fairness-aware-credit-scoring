"use strict";
(() => {
  const statusEl = document.getElementById("status");
  let saved = null;
  try { saved = JSON.parse(sessionStorage.getItem("lastResult") || "null"); } catch (e) { saved = null; }
  if (!saved) { document.getElementById("empty").hidden = false; return; }
  document.getElementById("content").hidden = false;

  const p = saved.predict;
  document.getElementById("prob").textContent = (p.probability * 100).toFixed(1) + "%";
  const band = document.getElementById("band");
  band.textContent = p.risk_band + " risk"; band.className = "band " + p.risk_band;
  document.getElementById("bandnote").textContent =
    "Risk band: Low means below the average default rate in the training data, High means the score is in the top 10% flagged by the model, Medium is in between. Model " + p.model_version + ".";
  const ul = document.getElementById("warnings");
  (p.warnings || []).forEach((w) => { const li = document.createElement("li"); li.textContent = w; ul.appendChild(li); });
  Charts.shapBars(document.getElementById("shap"), saved.explain.display || saved.explain.contributions);

  const out = document.getElementById("matched");
  function row(label, value) {
    const d = document.createElement("div");
    const b = document.createElement("strong"); b.textContent = label + ": ";
    d.appendChild(b); d.appendChild(document.createTextNode(value)); out.appendChild(d);
  }
  document.getElementById("match").addEventListener("click", async () => {
    const g = document.getElementById("gender").value;
    out.replaceChildren();
    if (!g) { showStatus(statusEl, "err", "Choose a gender to select the comparison pool."); return; }
    showStatus(statusEl, "info", "Searching for a matched applicant...");
    try {
      const r = await API.post("/matched-pair", { record: saved.record, gender: g });
      showStatus(statusEl, "", "");
      row("Your estimated probability", (r.own_probability * 100).toFixed(2) + "%");
      row("Matched applicant (other gender)", (r.matched_probability * 100).toFixed(2) + "%");
      row("Difference", (r.delta * 100).toFixed(2) + " percentage points");
      row("Typical difference across all matched pairs", (r.population_mean_abs_delta * 100).toFixed(2) + " percentage points");
      if (r.self_excluded) {
        const note = document.createElement("p");
        note.className = "muted small";
        note.textContent = "This applicant's own record is in the reference pool, so it was left out of the search. Sample applicants have no stated gender, so this comparison relies on the gender you chose.";
        out.appendChild(note);
      }
    } catch (e) {
      const msg = e.code === "NO_MATCH" ? "No sufficiently similar applicant was found." : e.message;
      showStatus(statusEl, "err", msg);
    }
  });
})();
