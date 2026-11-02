"use strict";
// Page 1: the form is built from GET /meta/options (nothing about fields or ranges is hard-coded here).
(() => {
  const statusEl = document.getElementById("status");
  const fieldsEl = document.getElementById("fields");
  const form = document.getElementById("applicant");
  let meta = null;

  const LABELS = FieldLabels.form;

  function el(tag, attrs, children) {
    const e = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => e.setAttribute(k, v));
    (children || []).forEach((c) => e.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return e;
  }

  function addField(name, control, hint) {
    const lab = el("label", { for: name }, [LABELS[name] || name]);
    if (hint) { const h = el("span", { class: "hint" }, [" " + hint]); lab.appendChild(h); }
    const wrap = el("div", { class: "field", "data-field": name }, [lab, control, el("div", { class: "msg", id: "msg_" + name })]);
    fieldsEl.appendChild(wrap);
  }

  function build() {
    fieldsEl.replaceChildren();
    Object.entries(meta.numeric).forEach(([name, r]) => {
      const rng = r.max !== null ? (r.min !== null ? r.min : "") + " to " + r.max : "min " + r.min;
      const inp = el("input", { id: name, name, type: "number", inputmode: "decimal", step: r.integer ? "1" : "any", autocomplete: "off" });
      if (r.min !== null) inp.min = r.min;
      if (r.max !== null) inp.max = r.max;
      addField(name, inp, r.nullable ? "(optional, " + rng + ")" : "(" + rng + ")");
    });
    Object.entries(meta.categorical).forEach(([name, opts]) => {
      const sel = el("select", { id: name, name });
      const nullable = meta.nullable_categorical.includes(name);
      sel.appendChild(el("option", { value: "" }, [nullable ? "Not stated" : "Choose..."]));
      opts.forEach((o) => sel.appendChild(el("option", { value: o }, [o])));
      addField(name, sel, nullable ? "(optional)" : "");
    });
    Object.entries(meta.flags).forEach(([name, opts]) => {
      const sel = el("select", { id: name, name });
      sel.appendChild(el("option", { value: "" }, ["Choose..."]));
      opts.forEach((o) => sel.appendChild(el("option", { value: o }, [o === "Y" ? "Yes" : "No"])));
      addField(name, sel, "");
    });
  }

  function readForm() {
    const rec = {};
    Object.entries(meta.numeric).forEach(([name, r]) => {
      const v = document.getElementById(name).value.trim();
      rec[name] = v === "" ? null : (r.integer ? parseInt(v, 10) : parseFloat(v));
    });
    Object.keys(meta.categorical).concat(Object.keys(meta.flags)).forEach((name) => {
      const v = document.getElementById(name).value;
      rec[name] = v === "" ? null : v;
    });
    return rec;
  }

  function writeForm(rec) {
    Object.keys(meta.numeric).forEach((n) => { document.getElementById(n).value = rec[n] === null || rec[n] === undefined ? "" : rec[n]; });
    Object.keys(meta.categorical).concat(Object.keys(meta.flags)).forEach((n) => {
      const s = document.getElementById(n);
      const v = rec[n] === null || rec[n] === undefined ? "" : rec[n];
      if (v !== "" && ![...s.options].some((o) => o.value === v)) s.appendChild(el("option", { value: v }, [v]));
      s.value = v;
    });
  }

  function setError(name, text) {
    const wrap = fieldsEl.querySelector('[data-field="' + name + '"]');
    if (!wrap) return;
    wrap.classList.toggle("invalid", !!text);
    wrap.querySelector(".msg").textContent = text || "";
  }

  function clientValidate(rec) {
    let ok = true;
    Object.entries(meta.numeric).forEach(([name, r]) => {
      const v = rec[name]; setError(name, "");
      if (v === null) { if (!r.nullable) { setError(name, "Required."); ok = false; } return; }
      if (Number.isNaN(v)) { setError(name, "Enter a number."); ok = false; return; }
      if ((r.min !== null && (r.exclusive_min ? v <= r.min : v < r.min)) || (r.max !== null && v > r.max)) { setError(name, "Out of range."); ok = false; }
    });
    Object.keys(meta.categorical).concat(Object.keys(meta.flags)).forEach((name) => {
      setError(name, "");
      if (rec[name] === null && !meta.nullable_categorical.includes(name)) { setError(name, "Required."); ok = false; }
    });
    return ok;
  }

  async function init() {
    showStatus(statusEl, "info", "Loading form options...");
    try {
      meta = await API.get("/meta/options");
      build();
      showStatus(statusEl, "", "");
    } catch (e) {
      showStatus(statusEl, "err", e.code === "API_DOWN" ? "The scoring service is not reachable." : e.message, init);
    }
  }

  document.getElementById("example").addEventListener("click", async () => {
    try {
      const data = await API.get("/examples");
      const ex = data.examples[Math.floor(Math.random() * data.examples.length)];
      writeForm(ex);
      Object.keys(meta.numeric).concat(Object.keys(meta.categorical), Object.keys(meta.flags)).forEach((n) => setError(n, ""));
    } catch (e) { showStatus(statusEl, "err", e.message); }
  });

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const rec = readForm();
    if (!clientValidate(rec)) { showStatus(statusEl, "err", "Please fix the highlighted fields."); return; }
    const btn = document.getElementById("submit");
    btn.disabled = true;
    showStatus(statusEl, "info", "Scoring...");
    try {
      const [pred, expl] = await Promise.all([API.post("/predict", rec), API.post("/explain", rec, 30000)]);
      // browser-only: kept in sessionStorage so result.html can render; never sent to a server store
      sessionStorage.setItem("lastResult", JSON.stringify({ record: rec, predict: pred, explain: expl }));
      window.location.href = "result.html";
    } catch (e) {
      btn.disabled = false;
      if (e.status === 422) {
        (e.details || []).forEach((d) => setError(d.field, d.issue));
        showStatus(statusEl, "err", "The server rejected some fields.");
      } else {
        showStatus(statusEl, "err", e.code === "API_DOWN" ? "The scoring service is not reachable." : e.message, () => form.requestSubmit());
      }
    }
  });

  init();
})();
