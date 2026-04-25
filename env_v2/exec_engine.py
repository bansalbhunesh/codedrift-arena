"""Subprocess pytest runner — single source of ground truth for V2.

Design goals:
- Hermetic per-episode execution (timeouts, output cap).
- Rich structured failure data (nodeid, file, line, exception, traceback).
- Optional ``pytest-json-report`` path; falls back to text parsing so the
  V2 stack runs on a vanilla pytest install.

Public surface: :func:`run_pytest` returning :class:`ExecutionResult`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_TIMEOUT_S = 30
MAX_OUTPUT_BYTES = 256_000


@dataclass
class FailedTest:
    nodeid: str
    file: str
    line: int
    exception: str
    message: str
    traceback: str
    call_chain: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    returncode: int
    duration_s: float
    passed: int
    failed: int
    errors: int
    failed_tests: list[FailedTest]
    stdout_tail: str
    stderr_tail: str
    timed_out: bool = False
    used_json_report: bool = False

    def as_dict(self) -> dict:
        return {
            "returncode": self.returncode,
            "duration_s": round(self.duration_s, 4),
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "timed_out": self.timed_out,
            "used_json_report": self.used_json_report,
            "failed_tests": [asdict(t) for t in self.failed_tests],
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


def _truncate(text: str, limit: int = MAX_OUTPUT_BYTES) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n... [truncated {len(text) - limit} bytes] ...\n{tail}"


def _has_pytest_json_report() -> bool:
    try:
        import pytest_jsonreport  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


_NODEID_RE = re.compile(r"FAILED\s+(\S+)\s*-\s*(.*)")
_ERROR_NODEID_RE = re.compile(r"^ERROR\s+(\S+)(?:\s+-\s+(.*))?$", re.MULTILINE)
_ERROR_COLLECT_RE = re.compile(
    r"ERROR collecting (\S+)[\s\S]*?(?=\n=+|\Z)", re.MULTILINE
)
_TRACEBACK_HEADER_RE = re.compile(r"^_+\s+(.*?)\s+_+$", re.MULTILINE)
_FRAME_PATH_RE = re.compile(r"^([^\n:]+\.py):(\d+):(.*)$", re.MULTILINE)


def _parse_text_output(stdout: str) -> tuple[int, int, int, list[FailedTest]]:
    """Best-effort parse of ``pytest -q --tb=short`` output.

    Counts and per-failure stanza extraction; richer structure comes from the
    json-report path when available.
    """
    passed = failed = errors = 0
    # Token-by-token summary scan handles both `=== 6 passed in 0.04s ===`
    # and the bare `6 passed in 0.04s` line emitted under `-q`.
    for kind, regex in (
        ("passed", r"(\d+)\s+passed"),
        ("failed", r"(\d+)\s+failed"),
        ("errors", r"(\d+)\s+errors?"),
    ):
        m = re.search(regex, stdout)
        if not m:
            continue
        n = int(m.group(1))
        if kind == "passed":
            passed = max(passed, n)
        elif kind == "failed":
            failed = max(failed, n)
        else:
            errors = max(errors, n)

    failed_nodes: dict[str, FailedTest] = {}
    for m in _NODEID_RE.finditer(stdout):
        nodeid = m.group(1).strip()
        message = m.group(2).strip()
        ft = failed_nodes.get(nodeid)
        if ft is None:
            failed_nodes[nodeid] = FailedTest(
                nodeid=nodeid,
                file=nodeid.split("::")[0] if "::" in nodeid else nodeid,
                line=0,
                exception=message.split(":", 1)[0].strip() if ":" in message else "Error",
                message=message,
                traceback="",
            )

    # Collection-time ERROR entries (e.g. ImportError when a renamed/removed
    # symbol is still referenced by tests). pytest emits both:
    #   ERRORS section with traceback, and
    #   `ERROR <nodeid>` summary line.
    for m in _ERROR_NODEID_RE.finditer(stdout):
        nodeid = m.group(1).strip()
        if nodeid in failed_nodes:
            continue
        message = (m.group(2) or "collection error").strip()
        failed_nodes[nodeid] = FailedTest(
            nodeid=nodeid,
            file=nodeid.split("::")[0] if "::" in nodeid else nodeid,
            line=0,
            exception="CollectionError",
            message=message,
            traceback="",
        )
    for m in _ERROR_COLLECT_RE.finditer(stdout):
        section = m.group(0)
        nodeid_m = re.search(r"ERROR collecting (\S+)", section)
        if not nodeid_m:
            continue
        nodeid = nodeid_m.group(1).strip()
        ft = failed_nodes.get(nodeid)
        exc_m = re.search(r"^E\s+(\w+(?:Error|Exception)):\s*(.*)$", section, re.MULTILINE)
        exception = exc_m.group(1).strip() if exc_m else "CollectionError"
        message = exc_m.group(2).strip() if exc_m else "import-time error"
        traceback = section[:4000]
        if ft is None:
            failed_nodes[nodeid] = FailedTest(
                nodeid=nodeid,
                file=nodeid,
                line=0,
                exception=exception,
                message=message,
                traceback=traceback,
                call_chain=[
                    f"{p.strip()}:{ln.strip()}" for p, ln, _ in _FRAME_PATH_RE.findall(section)
                ][-6:],
            )
        else:
            ft.exception = exception
            ft.message = message
            ft.traceback = traceback
            ft.call_chain = [
                f"{p.strip()}:{ln.strip()}" for p, ln, _ in _FRAME_PATH_RE.findall(section)
            ][-6:]

    # Walk per-failure stanzas to grab tracebacks and call chains.
    for m in _TRACEBACK_HEADER_RE.finditer(stdout):
        title = m.group(1).strip()
        start = m.end()
        end = stdout.find("\n___", start)
        if end == -1:
            end = len(stdout)
        section = stdout[start:end]
        nodeid_match = None
        for nodeid in failed_nodes:
            short = nodeid.split("::")[-1]
            if short and short in title:
                nodeid_match = nodeid
                break
        if nodeid_match is None:
            continue
        ft = failed_nodes[nodeid_match]
        ft.traceback = _truncate(section.strip(), 4000)
        frames = _FRAME_PATH_RE.findall(section)
        ft.call_chain = [f"{p.strip()}:{ln.strip()}" for p, ln, _ in frames][-6:]
        if frames and not ft.line:
            try:
                ft.line = int(frames[-1][1])
            except ValueError:
                ft.line = 0

    return passed, failed, errors, list(failed_nodes.values())


def _parse_json_report(report_path: Path) -> tuple[int, int, int, list[FailedTest]]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    passed = int(summary.get("passed", 0))
    failed = int(summary.get("failed", 0))
    errors = int(summary.get("error", 0)) + int(summary.get("errors", 0))
    failed_tests: list[FailedTest] = []
    for test in data.get("tests", []):
        outcome = test.get("outcome")
        if outcome not in {"failed", "error"}:
            continue
        nodeid = str(test.get("nodeid", ""))
        call = test.get("call") or {}
        crash = call.get("crash") or {}
        traceback_frames = call.get("traceback") or []
        call_chain = [
            f"{frame.get('path', '?')}:{frame.get('lineno', '?')}"
            for frame in traceback_frames[-6:]
        ]
        longrepr = call.get("longrepr") or test.get("longrepr") or ""
        failed_tests.append(
            FailedTest(
                nodeid=nodeid,
                file=nodeid.split("::")[0] if "::" in nodeid else nodeid,
                line=int(crash.get("lineno", 0) or 0),
                exception=str(crash.get("message", "")).split(":", 1)[0] or "Error",
                message=str(crash.get("message", "")),
                traceback=_truncate(str(longrepr), 4000),
                call_chain=call_chain,
            )
        )
    return passed, failed, errors, failed_tests


def run_pytest(
    repo_dir: Path,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    extra_args: Optional[list[str]] = None,
) -> ExecutionResult:
    """Run pytest inside ``repo_dir`` and return a structured result."""
    repo_dir = Path(repo_dir).resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"repo_dir does not exist: {repo_dir}")

    use_json = _has_pytest_json_report()
    json_report_path: Optional[Path] = None
    cmd: list[str] = [sys.executable, "-m", "pytest", "-q", "--tb=short", "--maxfail=20"]
    if use_json:
        json_report_path = Path(tempfile.mkstemp(prefix="pytest_v2_", suffix=".json")[1])
        cmd += ["--json-report", f"--json-report-file={json_report_path}"]
    if extra_args:
        cmd += list(extra_args)
    cmd += [str(repo_dir / "tests")]

    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "0")

    started = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode = int(proc.returncode)
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        returncode = -1
        timed_out = True
    duration = time.perf_counter() - started

    passed = failed = errors = 0
    failed_tests: list[FailedTest] = []
    used_json = False
    if use_json and json_report_path and json_report_path.exists() and json_report_path.stat().st_size > 0:
        try:
            passed, failed, errors, failed_tests = _parse_json_report(json_report_path)
            used_json = True
        except Exception:
            used_json = False
        try:
            json_report_path.unlink(missing_ok=True)
        except Exception:
            pass

    if not used_json:
        passed, failed, errors, failed_tests = _parse_text_output(stdout)

    return ExecutionResult(
        returncode=returncode,
        duration_s=duration,
        passed=passed,
        failed=failed,
        errors=errors,
        failed_tests=failed_tests,
        stdout_tail=_truncate(stdout, 16_000),
        stderr_tail=_truncate(stderr, 8_000),
        timed_out=timed_out,
        used_json_report=used_json,
    )
