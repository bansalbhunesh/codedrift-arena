"""Server middleware safety checks."""

from __future__ import annotations

import importlib
import json
import os
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestServerSecurity(unittest.TestCase):
    @staticmethod
    def _reset_server_module():
        # Prometheus REGISTRY is a process-wide singleton; metrics defined at
        # module scope in server.app stay registered after sys.modules.pop and
        # raise "Duplicated timeseries" on the next import. Unregister all
        # collectors so the reimport starts from a clean slate.
        if "server.app" in sys.modules:
            del sys.modules["server.app"]
        from prometheus_client import REGISTRY
        for collector in list(REGISTRY._collector_to_names):
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass

    def _load_app(self):
        self._reset_server_module()
        mod = importlib.import_module("server.app")
        return mod.app

    def test_health_is_public(self) -> None:
        os.environ["CODEDRIFT_API_TOKEN"] = "secret"
        try:
            app = self._load_app()
            client = TestClient(app)
            r = client.get("/health")
            self.assertEqual(r.status_code, 200)
            self.assertIn(r.json().get("status"), {"ok", "healthy"})
        finally:
            os.environ.pop("CODEDRIFT_API_TOKEN", None)

    def test_token_required_for_protected_routes(self) -> None:
        os.environ["CODEDRIFT_API_TOKEN"] = "secret"
        try:
            app = self._load_app()
            client = TestClient(app)
            r = client.get("/metadata")
            self.assertEqual(r.status_code, 401)
            self.assertTrue(r.json().get("request_id"))
            self.assertTrue(r.headers.get("x-request-id"))
        finally:
            os.environ.pop("CODEDRIFT_API_TOKEN", None)

    def test_payload_limit_blocks_oversized_request(self) -> None:
        os.environ["CODEDRIFT_API_MAX_BODY_BYTES"] = "32"
        os.environ["CODEDRIFT_API_TOKEN"] = ""
        try:
            app = self._load_app()
            client = TestClient(app)
            body = "x" * 500
            r = client.post("/step", content=body, headers={"content-type": "application/json"})
            self.assertEqual(r.status_code, 413)
            self.assertTrue(r.json().get("request_id"))
            self.assertTrue(r.headers.get("x-request-id"))
        finally:
            os.environ.pop("CODEDRIFT_API_MAX_BODY_BYTES", None)
            os.environ.pop("CODEDRIFT_API_TOKEN", None)

    def test_request_id_passthrough(self) -> None:
        os.environ["CODEDRIFT_API_TOKEN"] = ""
        try:
            app = self._load_app()
            client = TestClient(app)
            req_id = "req-demo-123"
            r = client.get("/health", headers={"x-request-id": req_id})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.headers.get("x-request-id"), req_id)
        finally:
            os.environ.pop("CODEDRIFT_API_TOKEN", None)

    def test_require_auth_without_token_fails_startup(self) -> None:
        os.environ["CODEDRIFT_REQUIRE_AUTH"] = "1"
        os.environ["CODEDRIFT_API_TOKEN"] = ""
        try:
            self._reset_server_module()
            with self.assertRaises(SystemExit):
                importlib.import_module("server.app")
        finally:
            os.environ.pop("CODEDRIFT_REQUIRE_AUTH", None)
            os.environ.pop("CODEDRIFT_API_TOKEN", None)

    def test_typed_api_reset_and_step(self) -> None:
        os.environ["CODEDRIFT_API_TOKEN"] = ""
        try:
            app = self._load_app()
            client = TestClient(app)
            r = client.post(
                "/api/v1/reset",
                json={"difficulty": "easy", "personality": "adaptive", "seed": 7},
            )
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertTrue(body.get("session_id"))
            self.assertTrue(body.get("observation", {}).get("prompt"))
            sid = body["session_id"]
            s = client.post(
                "/api/v1/step",
                json={
                    "session_id": sid,
                    "review": "VERDICT: REQUEST_CHANGES\nISSUES: getUserData stale\nREASON: mismatch.",
                },
            )
            self.assertEqual(s.status_code, 200)
            step_body = s.json()
            self.assertIn("reward", step_body)
            self.assertIn("info", step_body)
        finally:
            os.environ.pop("CODEDRIFT_API_TOKEN", None)

    def test_typed_api_unknown_session(self) -> None:
        os.environ["CODEDRIFT_API_TOKEN"] = ""
        try:
            app = self._load_app()
            client = TestClient(app)
            s = client.post(
                "/api/v1/step",
                json={
                    "session_id": "deadbeefdeadbeef.invalidsig",
                    "review": "VERDICT: APPROVE\nISSUES: none\nREASON: x",
                },
            )
            self.assertEqual(s.status_code, 403)
        finally:
            os.environ.pop("CODEDRIFT_API_TOKEN", None)

    def test_typed_api_session_single_use(self) -> None:
        os.environ["CODEDRIFT_API_TOKEN"] = ""
        try:
            app = self._load_app()
            client = TestClient(app)
            r = client.post("/api/v1/reset", json={"difficulty": "easy", "personality": "random", "seed": 1})
            self.assertEqual(r.status_code, 200)
            sid = r.json()["session_id"]
            body = {"session_id": sid, "review": "VERDICT: APPROVE\nISSUES: none\nREASON: x"}
            s1 = client.post("/api/v1/step", json=body)
            self.assertEqual(s1.status_code, 200)
            s2 = client.post("/api/v1/step", json=body)
            self.assertEqual(s2.status_code, 409)
        finally:
            os.environ.pop("CODEDRIFT_API_TOKEN", None)

    def test_typed_api_tampered_session_signature_rejected(self) -> None:
        os.environ["CODEDRIFT_API_TOKEN"] = ""
        try:
            app = self._load_app()
            client = TestClient(app)
            r = client.post("/api/v1/reset", json={"difficulty": "easy", "personality": "random", "seed": 2})
            self.assertEqual(r.status_code, 200)
            sid = r.json()["session_id"]
            raw, _sig = sid.split(".", 1)
            bad_sid = f"{raw}.deadbeefdeadbeef"
            s = client.post(
                "/api/v1/step",
                json={"session_id": bad_sid, "review": "VERDICT: APPROVE\nISSUES: none\nREASON: x"},
            )
            self.assertEqual(s.status_code, 403)
        finally:
            os.environ.pop("CODEDRIFT_API_TOKEN", None)

    def test_scope_read_token_cannot_write(self) -> None:
        os.environ["CODEDRIFT_API_READ_TOKEN"] = "readtok"
        os.environ["CODEDRIFT_API_WRITE_TOKEN"] = "writetok"
        try:
            app = self._load_app()
            client = TestClient(app)
            r = client.post(
                "/api/v1/reset",
                json={"difficulty": "easy", "personality": "random"},
                headers={"x-api-key": "readtok"},
            )
            self.assertEqual(r.status_code, 403)
            m = client.get("/metrics", headers={"x-api-key": "readtok"})
            self.assertIn(m.status_code, {200, 503})
        finally:
            os.environ.pop("CODEDRIFT_API_READ_TOKEN", None)
            os.environ.pop("CODEDRIFT_API_WRITE_TOKEN", None)

    def test_previous_signing_key_still_verifies(self) -> None:
        os.environ["CODEDRIFT_API_TOKEN"] = ""
        os.environ["CODEDRIFT_SESSION_SIGNING_KEY"] = "newkey"
        os.environ["CODEDRIFT_SESSION_PREVIOUS_SIGNING_KEYS"] = "oldkey"
        try:
            app = self._load_app()
            mod = importlib.import_module("server.app")
            client = TestClient(app)
            raw = "abcd1234abcd1234abcd1234abcd1234"
            import hashlib, hmac

            sig = hmac.new(b"oldkey", raw.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
            sid = f"{raw}.{sig}"
            now = __import__("time").time()
            session = {
                "schema_version": 2,
                "created_at": now,
                "expires_at": now + 900,
                "used": False,
                "episode_id": "ep123",
                "pr_diff": "diff --git a/x b/x",
                "actions": [],
            }
            mod._sessions[raw] = session
            s = client.post(
                "/api/v1/step",
                json={"session_id": sid, "review": "VERDICT: APPROVE\nISSUES: none\nREASON: x"},
            )
            self.assertEqual(s.status_code, 200)
        finally:
            os.environ.pop("CODEDRIFT_API_TOKEN", None)
            os.environ.pop("CODEDRIFT_SESSION_SIGNING_KEY", None)
            os.environ.pop("CODEDRIFT_SESSION_PREVIOUS_SIGNING_KEYS", None)

    def test_deprecated_schema_rejected(self) -> None:
        os.environ["CODEDRIFT_API_TOKEN"] = ""
        os.environ["CODEDRIFT_SESSION_MIN_SUPPORTED_SCHEMA_VERSION"] = "2"
        try:
            app = self._load_app()
            mod = importlib.import_module("server.app")
            client = TestClient(app)
            raw = "depr1234depr1234depr1234depr1234"
            sid = mod._sign_session_raw_id(raw)
            now = __import__("time").time()
            mod._sessions[raw] = {
                "schema_version": 1,
                "created_at": now,
                "expires_at": now + 900,
                "used": False,
                "episode_id": "ep_old",
                "pr_diff": "diff --git a/x b/x",
                "actions": [],
            }
            s = client.post(
                "/api/v1/step",
                json={"session_id": sid, "review": "VERDICT: APPROVE\nISSUES: none\nREASON: x"},
            )
            self.assertEqual(s.status_code, 409)
            self.assertIn("deprecated session schema", s.json().get("detail", ""))
        finally:
            os.environ.pop("CODEDRIFT_API_TOKEN", None)
            os.environ.pop("CODEDRIFT_SESSION_MIN_SUPPORTED_SCHEMA_VERSION", None)

    def test_api_reset_invalid_difficulty_returns_422(self) -> None:
        os.environ["CODEDRIFT_API_TOKEN"] = ""
        try:
            app = self._load_app()
            client = TestClient(app)
            r = client.post("/api/v1/reset", json={"difficulty": "nope", "personality": "random"})
            self.assertEqual(r.status_code, 422)
        finally:
            os.environ.pop("CODEDRIFT_API_TOKEN", None)

    def test_api_step_validation_error_returns_422(self) -> None:
        os.environ["CODEDRIFT_API_TOKEN"] = ""
        try:
            app = self._load_app()
            client = TestClient(app)
            r = client.post("/api/v1/step", json={"session_id": "x", "review": "ok"})
            self.assertEqual(r.status_code, 422)
        finally:
            os.environ.pop("CODEDRIFT_API_TOKEN", None)

    def test_in_memory_session_cap_evicts_oldest(self) -> None:
        os.environ["CODEDRIFT_API_TOKEN"] = ""
        os.environ["CODEDRIFT_MAX_IN_MEMORY_SESSIONS"] = "2"
        try:
            app = self._load_app()
            client = TestClient(app)
            r1 = client.post("/api/v1/reset", json={"difficulty": "easy", "personality": "random", "seed": 1})
            r2 = client.post("/api/v1/reset", json={"difficulty": "easy", "personality": "random", "seed": 2})
            r3 = client.post("/api/v1/reset", json={"difficulty": "easy", "personality": "random", "seed": 3})
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r2.status_code, 200)
            self.assertEqual(r3.status_code, 200)
            sid1 = r1.json()["session_id"]
            s = client.post(
                "/api/v1/step",
                json={"session_id": sid1, "review": "VERDICT: APPROVE\nISSUES: none\nREASON: x"},
            )
            self.assertEqual(s.status_code, 404)
        finally:
            os.environ.pop("CODEDRIFT_API_TOKEN", None)
            os.environ.pop("CODEDRIFT_MAX_IN_MEMORY_SESSIONS", None)

    def test_metrics_access_read_mode_requires_token(self) -> None:
        os.environ["CODEDRIFT_METRICS_ACCESS"] = "read"
        os.environ["CODEDRIFT_API_READ_TOKEN"] = "readtok"
        os.environ["CODEDRIFT_API_WRITE_TOKEN"] = "writetok"
        try:
            app = self._load_app()
            client = TestClient(app)
            r1 = client.get("/metrics")
            self.assertEqual(r1.status_code, 401)
            r2 = client.get("/metrics", headers={"x-api-key": "readtok"})
            self.assertIn(r2.status_code, {200, 503})
        finally:
            os.environ.pop("CODEDRIFT_METRICS_ACCESS", None)
            os.environ.pop("CODEDRIFT_API_READ_TOKEN", None)
            os.environ.pop("CODEDRIFT_API_WRITE_TOKEN", None)

