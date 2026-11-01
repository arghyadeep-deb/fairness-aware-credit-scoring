"use strict";
// Fetch wrapper: base URL, timeouts, error-body parsing. No applicant data is stored here.
const API = (() => {
  const base = "";
  async function call(method, path, body, timeoutMs) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs || 15000);
    try {
      const res = await fetch(base + path, {
        method,
        headers: body ? { "Content-Type": "application/json" } : {},
        body: body ? JSON.stringify(body) : undefined,
        signal: ctl.signal,
      });
      let data = null;
      try { data = await res.json(); } catch (e) { data = null; }
      if (!res.ok) {
        const err = (data && data.error) || { code: "HTTP_" + res.status, message: "Request failed", details: [] };
        const e = new Error(err.message);
        e.status = res.status; e.code = err.code; e.details = err.details || [];
        throw e;
      }
      return data;
    } catch (e) {
      if (e.name === "AbortError") { const t = new Error("The request timed out."); t.code = "TIMEOUT"; throw t; }
      if (e instanceof TypeError) { const n = new Error("Cannot reach the server."); n.code = "API_DOWN"; throw n; }
      throw e;
    } finally { clearTimeout(timer); }
  }
  return {
    get: (p, t) => call("GET", p, null, t),
    post: (p, b, t) => call("POST", p, b, t),
  };
})();

function showStatus(el, kind, text, retry) {
  el.replaceChildren();
  if (!text) return;
  const d = document.createElement("div");
  d.className = "banner " + kind;
  if (kind === "info") { const s = document.createElement("span"); s.className = "spinner"; d.appendChild(s); }
  d.appendChild(document.createTextNode(text));
  if (retry) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "secondary"; b.textContent = "Retry"; b.style.marginLeft = "12px";
    b.addEventListener("click", retry); d.appendChild(b);
  }
  el.appendChild(d);
}
