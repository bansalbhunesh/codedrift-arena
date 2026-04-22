"""
OpenEnv FastAPI entrypoint for CodeDrift Arena.

Run:
  uvicorn server.app:app --host 0.0.0.0 --port 8000

Requires: pip install -r requirements-server.txt

For a full reset→step episode over HTTP, use the **WebSocket** ``/ws`` session
(stateless ``POST /reset`` + ``POST /step`` each spin a new env in openenv-core).
Example: ``python scripts/openenv_ws_demo.py``
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import json
import hmac
import hashlib
from pathlib import Path
from threading import Lock

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from codedrift.logutil import get_logger
from codedrift.constants import DIFFICULTIES, PERSONALITIES
from env.codedrift_env import CodeDriftEnv
from agents.drift_agent import DriftAction
from rewards.scorer import RewardScorer
from integrations.codedrift_openenv import OPENENV_AVAILABLE, build_openenv_app

if not OPENENV_AVAILABLE:
    raise SystemExit(
        "openenv-core is not installed. Run: pip install 'openenv-core' uvicorn fastapi"
    )

app = build_openenv_app()
log = get_logger(__name__)

_MAX_BODY_BYTES = int(os.environ.get("CODEDRIFT_API_MAX_BODY_BYTES", str(256 * 1024)))
_RATE_LIMIT_RPM = int(os.environ.get("CODEDRIFT_API_RATE_LIMIT_RPM", "120"))
_AUTH_TOKEN = os.environ.get("CODEDRIFT_API_TOKEN", "").strip()
_REQUIRE_AUTH = os.environ.get("CODEDRIFT_REQUIRE_AUTH", "0").strip() == "1"
_READ_TOKEN = os.environ.get("CODEDRIFT_API_READ_TOKEN", "").strip()
_WRITE_TOKEN = os.environ.get("CODEDRIFT_API_WRITE_TOKEN", "").strip()
_REDIS_URL = os.environ.get("CODEDRIFT_REDIS_URL", "").strip()
_SESSION_TTL_SECONDS = int(os.environ.get("CODEDRIFT_SESSION_TTL_SECONDS", "900"))
_SESSION_SIGNING_KEY = os.environ.get("CODEDRIFT_SESSION_SIGNING_KEY", "dev-insecure-change-me").encode(
    "utf-8"
)
_SESSION_PREVIOUS_SIGNING_KEYS = [
    k.strip().encode("utf-8")
    for k in os.environ.get("CODEDRIFT_SESSION_PREVIOUS_SIGNING_KEYS", "").split(",")
    if k.strip()
]
_SESSION_SCHEMA_VERSION = 2
_SESSION_MIN_SUPPORTED_SCHEMA_VERSION = int(
    os.environ.get("CODEDRIFT_SESSION_MIN_SUPPORTED_SCHEMA_VERSION", "1")
)
_METRICS_ACCESS = os.environ.get("CODEDRIFT_METRICS_ACCESS", "public").strip().lower()
_WINDOW_SECONDS = 60
_hits_lock = Lock()
_hits: dict[str, tuple[int, int]] = {}
_redis_client = None
_sessions_lock = Lock()
_sessions: dict[str, dict[str, object]] = {}
_scorer = RewardScorer()

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    _METRICS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _METRICS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    Counter = Histogram = None  # type: ignore[assignment]
    generate_latest = None  # type: ignore[assignment]

if _METRICS_AVAILABLE:
    REQ_TOTAL = Counter("codedrift_api_requests_total", "Total API requests", ["path", "method", "status"])
    REQ_LATENCY = Histogram("codedrift_api_request_latency_seconds", "API request latency", ["path", "method"])
    RATE_LIMIT_HITS = Counter("codedrift_api_rate_limit_hits_total", "Rate-limit rejections")
    SCORE_LATENCY_MS = Histogram("codedrift_score_elapsed_ms", "Scorer elapsed time (ms)")
    AUTH_FAILURES = Counter(
        "codedrift_api_auth_failures_total",
        "Authentication/authorization failures",
        ["path", "reason"],
    )
    SESSION_REJECTIONS = Counter(
        "codedrift_api_session_rejections_total",
        "Session validation and schema rejections",
        ["reason"],
    )
else:
    REQ_TOTAL = REQ_LATENCY = RATE_LIMIT_HITS = SCORE_LATENCY_MS = AUTH_FAILURES = SESSION_REJECTIONS = None

if _REDIS_URL:
    try:
        import redis  # type: ignore

        _redis_client = redis.Redis.from_url(_REDIS_URL, decode_responses=True)
        _redis_client.ping()
        log.info("redis_rate_limiter_enabled")
    except Exception:
        _redis_client = None
        log.exception("redis_rate_limiter_init_failed_falling_back_to_memory")


def _validate_settings() -> None:
    if _MAX_BODY_BYTES <= 0:
        raise SystemExit("CODEDRIFT_API_MAX_BODY_BYTES must be > 0")
    if _RATE_LIMIT_RPM <= 0:
        raise SystemExit("CODEDRIFT_API_RATE_LIMIT_RPM must be > 0")
    if _SESSION_TTL_SECONDS <= 0:
        raise SystemExit("CODEDRIFT_SESSION_TTL_SECONDS must be > 0")
    if not _SESSION_SIGNING_KEY:
        raise SystemExit("CODEDRIFT_SESSION_SIGNING_KEY must be non-empty")
    if _SESSION_MIN_SUPPORTED_SCHEMA_VERSION <= 0:
        raise SystemExit("CODEDRIFT_SESSION_MIN_SUPPORTED_SCHEMA_VERSION must be > 0")
    if _SESSION_MIN_SUPPORTED_SCHEMA_VERSION > _SESSION_SCHEMA_VERSION:
        raise SystemExit("SESSION_MIN_SUPPORTED_SCHEMA_VERSION cannot exceed SESSION_SCHEMA_VERSION")
    if _METRICS_ACCESS not in {"public", "read", "write"}:
        raise SystemExit("CODEDRIFT_METRICS_ACCESS must be one of: public, read, write")
    if _REQUIRE_AUTH and not (_AUTH_TOKEN or _READ_TOKEN or _WRITE_TOKEN):
        raise SystemExit(
            "CODEDRIFT_REQUIRE_AUTH=1 requires CODEDRIFT_API_TOKEN or scoped read/write tokens"
        )


_validate_settings()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client is None:
        return "unknown"
    return request.client.host or "unknown"


def _audit(event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    log.info("audit %s", payload)


def _new_request_id(request: Request) -> str:
    incoming = request.headers.get("x-request-id", "").strip()
    if incoming:
        return incoming[:128]
    return uuid.uuid4().hex


def _allow_request(ip: str) -> tuple[bool, int]:
    now = int(time.time())
    window = now // _WINDOW_SECONDS
    redis_key = f"codedrift:rl:{ip}:{window}"
    if _redis_client is not None:
        try:
            count = int(_redis_client.incr(redis_key))
            if count == 1:
                _redis_client.expire(redis_key, _WINDOW_SECONDS + 2)
            return count <= _RATE_LIMIT_RPM, count
        except Exception:
            log.exception("redis_rate_limit_failed_falling_back_to_memory")

    with _hits_lock:
        prev_window, count = _hits.get(ip, (window, 0))
        if prev_window != window:
            prev_window, count = window, 0
        count += 1
        _hits[ip] = (prev_window, count)
        return count <= _RATE_LIMIT_RPM, count


def _is_write_route(request: Request) -> bool:
    if request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
        return True
    path = request.url.path
    return path.startswith("/api/v1/step") or path.startswith("/api/v1/reset")


def _authorize_request(request: Request, request_id: str) -> JSONResponse | None:
    path = request.url.path
    if path in {"/health", "/docs", "/openapi.json", "/redoc"}:
        return None
    presented = request.headers.get("x-api-key", "").strip()

    # Scoped token mode: write required for mutating routes, read or write for read routes.
    if _READ_TOKEN or _WRITE_TOKEN:
        if _is_write_route(request):
            if not _WRITE_TOKEN or presented != _WRITE_TOKEN:
                if AUTH_FAILURES is not None:
                    AUTH_FAILURES.labels(path=path, reason="forbidden_write_scope").inc()
                return JSONResponse(
                    status_code=403,
                    content={"detail": "forbidden: write scope required", "request_id": request_id},
                    headers={"x-request-id": request_id},
                )
            return None
        if presented not in {t for t in (_READ_TOKEN, _WRITE_TOKEN) if t}:
            if AUTH_FAILURES is not None:
                AUTH_FAILURES.labels(path=path, reason="missing_or_invalid_token").inc()
            return JSONResponse(
                status_code=401,
                content={"detail": "unauthorized", "request_id": request_id},
                headers={"x-request-id": request_id},
            )
        return None

    # Legacy single-token mode.
    if _AUTH_TOKEN and presented != _AUTH_TOKEN:
        if AUTH_FAILURES is not None:
            AUTH_FAILURES.labels(path=path, reason="invalid_legacy_token").inc()
        return JSONResponse(
            status_code=401,
            content={"detail": "unauthorized", "request_id": request_id},
            headers={"x-request-id": request_id},
        )
    return None


def _authorize_metrics_access(request: Request, request_id: str) -> JSONResponse | None:
    if _METRICS_ACCESS == "public":
        return None
    presented = request.headers.get("x-api-key", "").strip()
    if _METRICS_ACCESS == "read":
        allowed = {t for t in (_READ_TOKEN, _WRITE_TOKEN, _AUTH_TOKEN) if t}
        if presented in allowed:
            return None
        if AUTH_FAILURES is not None:
            AUTH_FAILURES.labels(path="/metrics", reason="metrics_read_token_required").inc()
        return JSONResponse(
            status_code=401,
            content={"detail": "unauthorized", "request_id": request_id},
            headers={"x-request-id": request_id},
        )
    # write
    if (_WRITE_TOKEN and presented == _WRITE_TOKEN) or (_AUTH_TOKEN and presented == _AUTH_TOKEN):
        return None
    if AUTH_FAILURES is not None:
        AUTH_FAILURES.labels(path="/metrics", reason="metrics_write_token_required").inc()
    return JSONResponse(
        status_code=403,
        content={"detail": "forbidden: write scope required", "request_id": request_id},
        headers={"x-request-id": request_id},
    )


def _sign_session_raw_id(raw_id: str) -> str:
    sig = hmac.new(_SESSION_SIGNING_KEY, raw_id.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
    return f"{raw_id}.{sig}"


def _verify_and_extract_raw_session_id(signed_id: str) -> str | None:
    if "." not in signed_id:
        return None
    raw_id, sig = signed_id.split(".", 1)
    keys = [_SESSION_SIGNING_KEY, *_SESSION_PREVIOUS_SIGNING_KEYS]
    for key in keys:
        expected = hmac.new(key, raw_id.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
        if hmac.compare_digest(sig, expected):
            return raw_id
    return None


def _serialize_actions(actions: list[DriftAction]) -> list[dict[str, object]]:
    return [
        {
            "drift_type": a.drift_type,
            "stale_ref": a.stale_ref,
            "current_ref": a.current_ref,
            "metadata": a.metadata,
        }
        for a in actions
    ]


def _deserialize_actions(payload: list[dict[str, object]]) -> list[DriftAction]:
    return [
        DriftAction(
            drift_type=str(d["drift_type"]),
            stale_ref=str(d["stale_ref"]),
            current_ref=str(d["current_ref"]),
            metadata=dict(d.get("metadata") or {}),
        )
        for d in payload
    ]


def _purge_expired_sessions(now: float) -> None:
    with _sessions_lock:
        expired = [sid for sid, data in _sessions.items() if float(data.get("expires_at", 0.0)) <= now]
        for sid in expired:
            _sessions.pop(sid, None)


def _session_redis_key(session_id: str) -> str:
    return f"codedrift:session:{session_id}"


def _store_session(session_id: str, data: dict[str, object]) -> None:
    if _redis_client is not None:
        try:
            key = _session_redis_key(session_id)
            _redis_client.set(key, json.dumps(data), ex=_SESSION_TTL_SECONDS)
            return
        except Exception:
            log.exception("redis_session_store_failed_falling_back_to_memory")
    with _sessions_lock:
        _sessions[session_id] = data


def _load_session(session_id: str) -> dict[str, object] | None:
    if _redis_client is not None:
        try:
            raw = _redis_client.get(_session_redis_key(session_id))
            if not raw:
                return None
            data = dict(json.loads(raw))
            ver = int(data.get("schema_version", 1))
            if ver > _SESSION_SCHEMA_VERSION:
                data["_unsupported_schema"] = ver
            if ver < _SESSION_MIN_SUPPORTED_SCHEMA_VERSION:
                data["_deprecated_schema"] = ver
            return data
        except Exception:
            log.exception("redis_session_load_failed_falling_back_to_memory")
    now = time.time()
    _purge_expired_sessions(now)
    with _sessions_lock:
        data = _sessions.get(session_id)
        if not data:
            return None
        if float(data.get("expires_at", 0.0)) <= now:
            _sessions.pop(session_id, None)
            return None
        out = dict(data)
        ver = int(out.get("schema_version", 1))
        if ver > _SESSION_SCHEMA_VERSION:
            out["_unsupported_schema"] = ver
        if ver < _SESSION_MIN_SUPPORTED_SCHEMA_VERSION:
            out["_deprecated_schema"] = ver
        return out


def _save_session_update(session_id: str, data: dict[str, object]) -> None:
    if _redis_client is not None:
        try:
            _redis_client.set(_session_redis_key(session_id), json.dumps(data), ex=_SESSION_TTL_SECONDS)
            return
        except Exception:
            log.exception("redis_session_update_failed_falling_back_to_memory")
    with _sessions_lock:
        _sessions[session_id] = data


class ResetRequest(BaseModel):
    difficulty: str = Field(default="easy")
    personality: str = Field(default="random")
    seed: int | None = Field(default=None)


class StepRequest(BaseModel):
    session_id: str = Field(min_length=12, max_length=96)
    review: str = Field(min_length=1, max_length=24000)


class ApiObservation(BaseModel):
    prompt: str
    pr_diff: str
    codebase_context: str
    episode_step: int
    n_stale_refs: int
    episode_id: str


@app.get("/health")
def health() -> dict[str, str]:
    """Container/lb healthcheck endpoint."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    if not _METRICS_AVAILABLE:
        return Response("metrics unavailable (install prometheus_client)\n", media_type="text/plain", status_code=503)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/v1/reset")
def api_reset(payload: ResetRequest, request: Request) -> dict[str, object]:
    if payload.difficulty not in DIFFICULTIES:
        raise HTTPException(status_code=422, detail=f"difficulty must be one of {sorted(DIFFICULTIES)}")
    if payload.personality not in PERSONALITIES:
        raise HTTPException(status_code=422, detail=f"personality must be one of {sorted(PERSONALITIES)}")
    env = CodeDriftEnv(
        difficulty=payload.difficulty,
        personality=payload.personality,
        seed=payload.seed,
    )
    obs = env.reset()
    raw_session_id = uuid.uuid4().hex
    session_id = _sign_session_raw_id(raw_session_id)
    now = time.time()
    session_data: dict[str, object] = {
        "schema_version": _SESSION_SCHEMA_VERSION,
        "created_at": now,
        "expires_at": now + _SESSION_TTL_SECONDS,
        "used": False,
        "episode_id": env.episode_id,
        "pr_diff": env.pr_diff,
        "actions": _serialize_actions(env.stale_actions),
    }
    _store_session(raw_session_id, session_data)
    request_id = getattr(request.state, "request_id", "")
    _audit(
        "api_reset",
        request_id=request_id,
        session_id=session_id,
        episode_id=env.episode_id,
        difficulty=payload.difficulty,
        personality=payload.personality,
    )
    return {
        "session_id": session_id,
        "observation": ApiObservation(
            prompt=obs.prompt,
            pr_diff=obs.pr_diff,
            codebase_context=obs.codebase_context,
            episode_step=obs.episode_step,
            n_stale_refs=obs.n_stale_refs,
            episode_id=env.episode_id,
        ).model_dump(),
    }


@app.post("/api/v1/step")
def api_step(payload: StepRequest, request: Request) -> dict[str, object]:
    raw_session_id = _verify_and_extract_raw_session_id(payload.session_id)
    if raw_session_id is None:
        if SESSION_REJECTIONS is not None:
            SESSION_REJECTIONS.labels(reason="invalid_signature").inc()
        raise HTTPException(status_code=403, detail="invalid session signature")

    session = _load_session(raw_session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session_id; call /api/v1/reset first")
    if "_unsupported_schema" in session:
        if SESSION_REJECTIONS is not None:
            SESSION_REJECTIONS.labels(reason="unsupported_schema").inc()
        raise HTTPException(
            status_code=409,
            detail=f"unsupported session schema {session.get('_unsupported_schema')}",
        )
    if "_deprecated_schema" in session:
        if SESSION_REJECTIONS is not None:
            SESSION_REJECTIONS.labels(reason="deprecated_schema").inc()
        raise HTTPException(
            status_code=409,
            detail=f"deprecated session schema {session.get('_deprecated_schema')}",
        )
    if bool(session.get("used")):
        raise HTTPException(status_code=409, detail="session already consumed; call /api/v1/reset")
    actions_payload = session.get("actions")
    pr_diff = str(session.get("pr_diff", ""))
    if not isinstance(actions_payload, list):
        raise HTTPException(status_code=500, detail="corrupt session payload")
    actions = _deserialize_actions(actions_payload)
    reward, info = _scorer.score(payload.review, actions, pr_diff)
    info.setdefault("episode_id", str(session.get("episode_id", "")))
    done = True
    session["used"] = True
    session["expires_at"] = time.time() + _SESSION_TTL_SECONDS
    _save_session_update(raw_session_id, session)
    request_id = getattr(request.state, "request_id", "")
    if SCORE_LATENCY_MS is not None and "score_elapsed_ms" in info:
        SCORE_LATENCY_MS.observe(float(info["score_elapsed_ms"]))
    _audit(
        "api_step",
        request_id=request_id,
        session_id=raw_session_id,
        episode_id=info.get("episode_id"),
        reward=reward,
        outcome=info.get("episode_outcome"),
    )
    return {"reward": reward, "done": done, "info": info}


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    path = request.url.path
    request_id = _new_request_id(request)
    request.state.request_id = request_id
    started = time.perf_counter()
    if path in {"/health", "/docs", "/openapi.json", "/redoc"}:
        resp = await call_next(request)
        resp.headers["x-request-id"] = request_id
        if REQ_TOTAL is not None:
            REQ_TOTAL.labels(path=path, method=request.method, status=str(resp.status_code)).inc()
            REQ_LATENCY.labels(path=path, method=request.method).observe(time.perf_counter() - started)
        return resp
    if path == "/metrics":
        auth_err = _authorize_metrics_access(request, request_id)
        if auth_err is not None:
            return auth_err
        resp = await call_next(request)
        resp.headers["x-request-id"] = request_id
        if REQ_TOTAL is not None:
            REQ_TOTAL.labels(path=path, method=request.method, status=str(resp.status_code)).inc()
            REQ_LATENCY.labels(path=path, method=request.method).observe(time.perf_counter() - started)
        return resp

    auth_err = _authorize_request(request, request_id)
    if auth_err is not None:
        return auth_err

    # Request body cap to avoid memory abuse.
    content_len = request.headers.get("content-length")
    if content_len:
        try:
            if int(content_len) > _MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "payload too large", "request_id": request_id},
                    headers={"x-request-id": request_id},
                )
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"detail": "invalid content-length", "request_id": request_id},
                headers={"x-request-id": request_id},
            )

    # Global-ish limiter via Redis when configured; in-memory fallback otherwise.
    ip = _client_ip(request)
    allowed, count = _allow_request(ip)
    if not allowed:
        log.warning(
            "rate_limit_exceeded ip=%s path=%s count=%s request_id=%s",
            ip,
            path,
            count,
            request_id,
        )
        if RATE_LIMIT_HITS is not None:
            RATE_LIMIT_HITS.inc()
        return JSONResponse(
            status_code=429,
            content={"detail": "rate limit exceeded", "request_id": request_id},
            headers={"x-request-id": request_id},
        )
    resp = await call_next(request)
    resp.headers["x-request-id"] = request_id
    if REQ_TOTAL is not None:
        REQ_TOTAL.labels(path=path, method=request.method, status=str(resp.status_code)).inc()
        REQ_LATENCY.labels(path=path, method=request.method).observe(time.perf_counter() - started)
    return resp
