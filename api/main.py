"""FastAPI app: stateless scoring from the frozen bundle plus the static frontend (one process).

    uvicorn api.main:app --port 8000        # API + http://127.0.0.1:8000/

No database, no file writes, no applicant data in logs (request id, endpoint, status, latency only).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from api import keepalive
from api.bundle import Bundle, load_bundle
from api.explain import explain
from api.matched import matched_pair
from api.schemas import (
    ApplicantRecord,
    ExplainResponse,
    MatchedRequest,
    MatchedResponse,
    PredictResponse,
)
from api.service import predict

log = logging.getLogger("api")
WEB = Path(__file__).resolve().parents[1] / "web"
SYNTHETIC = Path(__file__).resolve().parent / "data" / "synthetic_examples.json"
NUMERIC_FIELDS = [n for n in ApplicantRecord.model_fields if not n.startswith("name_") and n not in
                  ("occupation_type", "organization_type", "flag_own_car", "flag_own_realty")]  # fmt: skip


def error(status: int, code: str, message: str, details: list | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "details": details or []}})  # fmt: skip


def create_app(bundle_dir: str | Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        torch.set_num_threads(2)
        app.state.bundle, app.state.error = None, None
        try:
            app.state.bundle = load_bundle(bundle_dir)
            log.info("bundle loaded: %s", app.state.bundle.sha[:12])
        except Exception as exc:  # fail closed: every data endpoint answers 503
            app.state.error = f"{type(exc).__name__}: {exc}"
            log.error("bundle NOT loaded: %s", app.state.error)
        ka = keepalive.start()  # off unless KEEPALIVE_URL or RENDER_EXTERNAL_URL is set
        yield
        if ka is not None:
            ka.cancel()

    app = FastAPI(title="Fairness-aware credit scoring (research prototype)", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        rid, t0 = uuid.uuid4().hex[:8], time.perf_counter()
        resp = await call_next(request)
        # request id, endpoint, status, latency only; never bodies or applicant values
        log.info("req=%s %s %s -> %s %.1fms", rid, request.method, request.url.path, resp.status_code,
                 (time.perf_counter() - t0) * 1000)  # fmt: skip
        resp.headers["X-Request-Id"] = rid
        return resp

    def need(request: Request) -> Bundle | JSONResponse:
        b = request.app.state.bundle
        if b is None:
            return error(503, "MODEL_NOT_READY", "Model bundle failed to load or verify.")
        return b

    @app.exception_handler(RequestValidationError)
    async def on_validation(request: Request, exc: RequestValidationError):
        det = [
            {"field": ".".join(str(x) for x in e["loc"] if x != "body"), "issue": e["msg"]}
            for e in exc.errors()
        ]
        if any(e["type"] == "json_invalid" for e in exc.errors()) or any(
            (request.url.path == "/matched-pair" and d["field"] == "gender") for d in det
        ):
            return error(400, "BAD_REQUEST", "Malformed JSON or invalid gender value.", det)
        return error(422, "VALIDATION_ERROR", "Invalid request fields.", det)

    @app.exception_handler(StarletteHTTPException)
    async def on_http(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404:
            return error(404, "NOT_FOUND", "Unknown route.")
        return error(exc.status_code, "BAD_REQUEST", str(exc.detail))

    @app.exception_handler(Exception)
    async def on_error(request: Request, exc: Exception):
        log.exception("internal error")  # traceback to the server log only, no request values
        return error(500, "INTERNAL_ERROR", "Unexpected error.")

    @app.get("/health")
    def health(request: Request):
        b = request.app.state.bundle
        if b is None:
            return error(503, "MODEL_NOT_READY", "Model bundle failed to load or verify.")
        return {"status": "ok", "model_version": b.version, "bundle_sha": b.sha}

    @app.post("/predict", response_model=PredictResponse)
    def post_predict(rec: ApplicantRecord, request: Request):
        b = need(request)
        return b if isinstance(b, JSONResponse) else predict(b, rec)

    @app.post("/explain", response_model=ExplainResponse)
    def post_explain(rec: ApplicantRecord, request: Request):
        b = need(request)
        return b if isinstance(b, JSONResponse) else explain(b, rec)

    @app.post("/matched-pair", response_model=MatchedResponse)
    def post_matched(req: MatchedRequest, request: Request):
        b = need(request)
        return b if isinstance(b, JSONResponse) else matched_pair(b, req.record, req.gender)

    def doc(name: str):
        def handler(request: Request):
            b = need(request)
            return (
                b if isinstance(b, JSONResponse) else {**b.docs[name], "model_version": b.version}
            )

        return handler

    for route, name in (("/fairness", "fairness"), ("/calibration", "calibration"),
                        ("/proxy-audit", "proxy_audit"), ("/comparison", "comparison")):  # fmt: skip
        app.get(route)(doc(name))

    @app.get("/tradeoff")
    def tradeoff(
        request: Request, lambda_f: float = 0.05, lambda_c: float = 2.0, lambda_a: float = 0.0
    ):
        b = need(request)
        if isinstance(b, JSONResponse):
            return b
        t = b.docs["tradeoff"]
        pts = t["points"]
        scale = [max(p[k] for p in pts) or 1.0 for k in ("lambda_f", "lambda_c", "lambda_a")]
        req = (lambda_f, lambda_c, lambda_a)

        def dist(p):
            return sum(
                ((p[k] - r) / s) ** 2
                for k, r, s in zip(("lambda_f", "lambda_c", "lambda_a"), req, scale)
            )

        near = min(pts, key=dist)
        return {"requested": {"lambda_f": lambda_f, "lambda_c": lambda_c, "lambda_a": lambda_a},
                "nearest": {k: near[k] for k in ("lambda_f", "lambda_c", "lambda_a")},
                "metrics": {k: near[k] for k in ("auc_roc", "auc_pr", "dir", "eod", "worst_group_ece")},
                "points": pts, "frontier": t["frontier"], "selected": t["selected"], "source": t["source"],
                "model_version": b.version}  # fmt: skip

    @app.get("/meta/options")
    def meta_options(request: Request):
        b = need(request)
        if isinstance(b, JSONResponse):
            return b
        from api.service import CAT_MAP

        fields = ApplicantRecord.model_fields
        numeric = {}
        for n in NUMERIC_FIELDS:
            lo = hi = None
            for m in fields[n].metadata:
                lo = getattr(m, "ge", lo) if hasattr(m, "ge") and m.ge is not None else lo
                lo = getattr(m, "gt", lo) if hasattr(m, "gt") and m.gt is not None else lo
                hi = m.le if hasattr(m, "le") and m.le is not None else hi
            numeric[n] = {"min": lo, "max": hi, "nullable": not fields[n].is_required(),
                          "integer": "int" in str(fields[n].annotation), "exclusive_min": "gt" in str(fields[n].metadata)}  # fmt: skip
        cats = {k: sorted(b.pre.encoders[col]) for k, col in CAT_MAP.items()}
        return {"categorical": cats, "numeric": numeric, "flags": {"flag_own_car": ["Y", "N"], "flag_own_realty": ["Y", "N"]},
                "nullable_categorical": ["occupation_type", "organization_type"],
                "risk_bands": b.manifest["risk_bands"], "include_ext_source": True, "model_version": b.version}  # fmt: skip

    @app.get("/examples")
    def examples(request: Request):
        b = need(request)
        if isinstance(b, JSONResponse):
            return b
        # Synthetic demo applicants (scripts/make_synthetic_examples.py) replace the real held-out rows
        # stored in the frozen bundle; fall back to the bundle only if the synthetic file is missing.
        if SYNTHETIC.is_file():
            syn = json.loads(SYNTHETIC.read_text(encoding="utf-8"))
            return {"examples": syn["examples"], "source": "synthetic", "model_version": b.version}
        return {"examples": b.examples, "source": "held-out rows", "model_version": b.version}

    if WEB.is_dir():
        app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    return app


app = create_app()
