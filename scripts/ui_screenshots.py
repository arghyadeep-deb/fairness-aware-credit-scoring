"""Phase 7 end-to-end UI check: start the server, drive headless Edge/Chrome over CDP, save screenshots.

python scripts/ui_screenshots.py            -> artifacts/ui/*.png and artifacts/ui/ui_check.json
Uses only local HTTP (127.0.0.1) plus the pinned Chart.js CDN URL the pages reference.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import websocket

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "ui"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

PORT = int(os.environ.get("UI_PORT", "8000"))
BASE, DBG = f"http://127.0.0.1:{PORT}", 9333


class CDP:
    def __init__(self, ws_url: str) -> None:
        self.ws = websocket.create_connection(ws_url, timeout=60, suppress_origin=True)
        self.n = 0

    def call(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr: str):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    def wait(self, cond: str, timeout: float = 40.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.js(cond):
                return True
            time.sleep(0.4)
        return False

    def shot(self, name: str, full: bool = True) -> str:
        h = self.js("Math.max(document.body.scrollHeight, 900)") if full else 900
        self.call("Emulation.setDeviceMetricsOverride", width=1280, height=int(min(h, 6000)), deviceScaleFactor=1, mobile=False)
        time.sleep(0.8)
        data = self.call("Page.captureScreenshot", format="png")["data"]
        p = OUT / name
        p.write_bytes(base64.b64decode(data))
        return p.name


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    browser = EDGE if Path(EDGE).exists() else CHROME
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(PORT), "--log-level", "warning"], cwd=ROOT)
    prof = ROOT / "artifacts" / "ui" / "_profile"
    report: dict = {"browser": Path(browser).name, "steps": {}}
    br = None
    try:
        for _ in range(60):
            try:
                if httpx.get(BASE + "/health", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("server did not start")
        br = subprocess.Popen([browser, "--headless=new", f"--remote-debugging-port={DBG}", f"--user-data-dir={prof}", "--no-first-run", "--disable-gpu", "about:blank"])
        ws = None
        for _ in range(40):
            try:
                tabs = httpx.get(f"http://127.0.0.1:{DBG}/json", timeout=2).json()
                ws = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.5)
        if not ws:
            raise RuntimeError("no devtools target")
        c = CDP(ws)
        c.call("Page.enable")
        # ---- page 1: form built from /meta/options, example loaded ----
        c.call("Page.navigate", url=BASE + "/index.html")
        report["steps"]["form_built"] = c.wait("document.querySelectorAll('#fields .field').length >= 18")
        c.js("document.getElementById('example').click()")
        time.sleep(1.5)
        report["steps"]["example_loaded"] = bool(c.js("document.getElementById('age_years').value !== ''"))
        report["steps"]["open_range_labels_ok"] = bool(c.js("!document.getElementById('fields').textContent.includes(' to )') && document.getElementById('fields').textContent.includes('min 0')"))
        c.js("document.getElementById('ext_source_3').value = ''")  # exercise the merged "not provided" SHAP label
        report["steps"]["screenshot_form"] = c.shot("01_form.png")
        # ---- submit -> result page ----
        c.js("document.getElementById('submit').click()")
        report["steps"]["result_page"] = c.wait("location.pathname.endsWith('result.html') && !document.getElementById('content').hidden", 90)
        time.sleep(2.5)
        report["steps"]["probability_text"] = c.js("document.getElementById('prob').textContent")
        report["steps"]["band_text"] = c.js("document.getElementById('band').textContent")
        report["steps"]["shap_chart_drawn"] = bool(c.js("typeof Chart !== 'undefined' && !!Chart.getChart('shap')"))
        labels = c.js("Chart.getChart('shap').data.labels") or []
        report["steps"]["shap_labels"] = labels
        report["steps"]["shap_labels_friendly"] = bool(labels) and not any("_" in str(x).split(" = ")[0] for x in labels)
        full = c.js("JSON.parse(sessionStorage.getItem('lastResult')).explain.display.filter(x => x.feature === 'EXT_SOURCE_3').map(x => FieldLabels.shapLabel(x))[0]")
        report["steps"]["missing_score_label"] = full
        c.js("document.getElementById('gender').value = 'F'; document.getElementById('match').click()")
        report["steps"]["matched_panel"] = c.wait("document.getElementById('matched').textContent.includes('Difference')", 40)
        report["steps"]["screenshot_result"] = c.shot("02_result.png")
        # ---- dashboard + slider ----
        c.call("Page.navigate", url=BASE + "/dashboard.html")
        report["steps"]["dashboard_loaded"] = c.wait("document.querySelectorAll('#cmp tr').length > 20", 60)
        time.sleep(2.0)
        report["steps"]["charts_drawn"] = bool(c.js("typeof Chart !== 'undefined' && ['fair','calib','trade'].every(id => !!Chart.getChart(id))"))
        c.js("var s=document.getElementById('lf'); s.value=0.25; s.dispatchEvent(new Event('input'));")
        time.sleep(1.5)
        report["steps"]["slider_nearest"] = c.js("document.getElementById('near').textContent.slice(0,120)")
        report["steps"]["honest_banner"] = c.js("document.getElementById('honest').textContent")
        report["steps"]["proxy_label_ok"] = bool(c.js("document.getElementById('proxy').textContent.includes('adversary off, not a GRL result')"))
        report["steps"]["main_table_rows"] = c.js("[...document.querySelectorAll('#cmp_main tr')].slice(1).map(r => r.cells[0].textContent)")
        report["steps"]["served_highlighted"] = c.js("document.querySelector('#cmp_main tr.served').cells[0].textContent")
        report["steps"]["ci_hover_sample"] = c.js("document.querySelector('#cmp_main tr.served td[title]').title")
        report["steps"]["cell_sample"] = c.js("document.querySelector('#cmp_main tr.served td[title]').textContent")
        c.js("document.getElementById('cmp_all').open = true")
        report["steps"]["full_table_rows"] = c.js("document.querySelectorAll('#cmp tr').length - 1")
        report["steps"]["no_row_cut_off"] = bool(c.js("[...document.querySelectorAll('.tablewrap.full')].every(w => w.scrollHeight <= w.clientHeight + 1)"))
        report["steps"]["screenshot_dashboard"] = c.shot("03_dashboard.png")
        # The top disclaimer banner was removed at the owner's request; check that every page still loads.
        report["steps"]["all_pages_load"] = True
        for page in ("index.html", "result.html", "dashboard.html"):
            c.call("Page.navigate", url=f"{BASE}/{page}")
            time.sleep(1.2)
            if not c.js("!!document.querySelector('header') && !document.querySelector('.disclaimer')"):
                report["steps"]["all_pages_load"] = False
        c.ws.close()
    finally:
        if br:
            br.terminate()
        server.terminate()
    (OUT / "ui_check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    ok = all(v for k, v in report["steps"].items() if k in ("form_built", "example_loaded", "result_page", "matched_panel", "dashboard_loaded"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
