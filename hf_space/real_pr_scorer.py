"""Multi-language Real-PR scoring helpers used by the Gradio Space.

What this module does:
- ``detect_languages(diff)``: best-effort language guess from ``diff --git`` paths
  and inline file extensions.
- ``extract_candidate_stale_refs(diff, languages)``: scans the diff's removed
  lines (``-`` lines) for symbols that disappeared from the post-diff source
  (``+`` lines) — a heuristic for renames / removals / signature changes.
- ``score_real_pr(...)``: rebuilds a synthetic episode using
  :meth:`env.codedrift_env.CodeDriftEnv.inject_episode` and scores the
  reviewer response against the stale refs the user confirms.

Caveats (we tell the user this in the UI):
- We do not run the PR's real test suite. The reward is based on whether the
  review's ``ISSUES`` block cites the stale refs that the user (or the
  detector) provided.
- Heuristic detection is intentionally conservative; users can edit the list
  before scoring.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from agents.drift_agent import DriftAction
from env.codebase import build_base_codebase
from env.codedrift_env import CodeDriftEnv

EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".swift": "swift",
    ".scala": "scala",
}

# Per-language regex patterns to find "symbols defined / used" on a line.
# These are intentionally simple — we want recall, not precision.
_LANG_SYMBOL_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "python": [
        re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
        re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
        re.compile(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import\s+"),
        re.compile(r"^\s*import\s+([A-Za-z_][\w.]*)"),
        re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    ],
    "javascript": [
        re.compile(r"^\s*function\s+([A-Za-z_$][\w$]*)\s*\("),
        re.compile(r"^\s*export\s+(?:default\s+)?function\s+([A-Za-z_$][\w$]*)\s*\("),
        re.compile(r"^\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*="),
        re.compile(r"^\s*class\s+([A-Za-z_$][\w$]*)\b"),
        re.compile(r"\b([A-Za-z_$][\w$]*)\s*\("),
        re.compile(r"^\s*import\s+(?:.*?from\s+)?['\"]([^'\"]+)['\"]"),
    ],
    "typescript": [
        re.compile(r"^\s*function\s+([A-Za-z_$][\w$]*)\s*[<(]"),
        re.compile(r"^\s*export\s+(?:default\s+)?function\s+([A-Za-z_$][\w$]*)\s*[<(]"),
        re.compile(r"^\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*[:=]"),
        re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)\b"),
        re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)\b"),
        re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\s*="),
        re.compile(r"\b([A-Za-z_$][\w$]*)\s*\("),
        re.compile(r"^\s*import\s+(?:.*?from\s+)?['\"]([^'\"]+)['\"]"),
    ],
    "go": [
        re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Z_][\w]*)\s*\("),
        re.compile(r"^\s*type\s+([A-Za-z_][\w]*)\b"),
        re.compile(r"\b([A-Za-z_][\w]*)\s*\("),
    ],
    "java": [
        re.compile(r"^\s*(?:public|private|protected|static|\s)+\s+[\w<>\[\],\s]+\s+([A-Za-z_][\w]*)\s*\("),
        re.compile(r"^\s*(?:public|private|protected|abstract|final|\s)*class\s+([A-Za-z_][\w]*)\b"),
        re.compile(r"\b([A-Za-z_][\w]*)\s*\("),
    ],
    "rust": [
        re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][\w]*)\s*[<(]"),
        re.compile(r"^\s*(?:pub\s+)?struct\s+([A-Za-z_][\w]*)\b"),
        re.compile(r"\b([A-Za-z_][\w]*)\s*\("),
    ],
    "c": [
        re.compile(r"^\s*[\w*\s]+\s+([A-Za-z_][\w]*)\s*\("),
        re.compile(r"^\s*#include\s+[<\"]([^>\"]+)[>\"]"),
    ],
    "cpp": [
        re.compile(r"^\s*[\w:*&\s]+\s+([A-Za-z_][\w]*)\s*\("),
        re.compile(r"^\s*class\s+([A-Za-z_][\w]*)\b"),
        re.compile(r"^\s*#include\s+[<\"]([^>\"]+)[>\"]"),
    ],
}

# Words we never want to treat as a stale ref (too generic / always present).
_NOISE_TOKENS = frozenset({
    "if", "else", "elif", "for", "while", "try", "except", "with", "return", "yield",
    "import", "from", "as", "in", "is", "not", "and", "or", "true", "false", "none",
    "this", "self", "new", "let", "var", "const", "function", "class", "type",
    "public", "private", "protected", "static", "void", "int", "str", "bool", "list",
    "dict", "tuple", "set", "default", "async", "await", "throw", "throws",
})


@dataclass
class DiffSummary:
    files: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0


def detect_languages(diff: str) -> DiffSummary:
    """Return per-file paths + best-effort language guesses for a unified diff."""
    summary = DiffSummary()
    seen_lang: set[str] = set()
    seen_file: set[str] = set()
    for line in diff.splitlines():
        if line.startswith("diff --git"):
            m = re.match(r"diff --git a/(\S+) b/(\S+)", line)
            if m:
                _add_file(summary, seen_file, seen_lang, m.group(2) or m.group(1))
        elif line.startswith("+++ ") and "b/" in line:
            path = line.split("b/", 1)[1].strip()
            if path and path != "/dev/null":
                _add_file(summary, seen_file, seen_lang, path)
        elif line.startswith("--- ") and "a/" in line:
            path = line.split("a/", 1)[1].strip()
            if path and path != "/dev/null":
                _add_file(summary, seen_file, seen_lang, path)
        elif line.startswith("+") and not line.startswith("+++"):
            summary.additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            summary.deletions += 1
    summary.languages = sorted(seen_lang)
    return summary


def _add_file(summary: DiffSummary, files: set[str], langs: set[str], path: str) -> None:
    if path in files:
        return
    files.add(path)
    summary.files.append(path)
    ext = path[path.rfind("."):].lower() if "." in path else ""
    lang = EXT_TO_LANG.get(ext)
    if lang:
        langs.add(lang)


def extract_candidate_stale_refs(diff: str, languages: Iterable[str]) -> list[str]:
    """Return candidate stale identifiers — symbols present on ``-`` lines but
    missing from ``+`` lines of the same file. Best-effort, language-aware."""
    if not diff.strip():
        return []
    add_lines: list[str] = []
    del_lines: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            add_lines.append(line[1:])
        elif line.startswith("-"):
            del_lines.append(line[1:])

    pattern_pool: list[re.Pattern[str]] = []
    seen = set()
    for lang in languages:
        for p in _LANG_SYMBOL_PATTERNS.get(lang, []):
            if p.pattern in seen:
                continue
            pattern_pool.append(p)
            seen.add(p.pattern)
    if not pattern_pool:
        # Generic fallback: any C-like identifier used as a callee.
        pattern_pool.append(re.compile(r"\b([A-Za-z_][\w]*)\s*\("))
        pattern_pool.append(re.compile(r"^\s*import\s+['\"]?([A-Za-z_][\w./-]*)['\"]?"))

    def _symbols(lines: list[str]) -> set[str]:
        out: set[str] = set()
        for line in lines:
            for pat in pattern_pool:
                for m in pat.finditer(line):
                    name = m.group(1).strip()
                    low = name.lower()
                    if not name or low in _NOISE_TOKENS:
                        continue
                    if len(name) <= 1:
                        continue
                    out.add(name)
        return out

    removed = _symbols(del_lines)
    kept = _symbols(add_lines)
    candidates = sorted(removed - kept)
    # Cap to avoid overwhelming the UI.
    return candidates[:20]


def score_real_pr(
    pr_diff: str,
    review: str,
    stale_refs: list[str],
    drift_kind: str = "rename",
) -> tuple[float, dict, dict]:
    """Score a real PR diff against user-provided stale refs.

    ``drift_kind`` selects the reward branch (``rename`` works for most
    real-world PRs because it just checks symbol mention; ``removal`` is for
    deleted modules; ``contract`` is for changed signatures).
    """
    if not pr_diff.strip():
        raise ValueError("pr_diff is empty")
    if not stale_refs:
        raise ValueError("provide at least one stale_ref to score against")

    actions: list[DriftAction] = []
    for ref in stale_refs:
        if drift_kind == "removal":
            actions.append(
                DriftAction(
                    drift_type="removal",
                    stale_ref=ref,
                    current_ref="[deleted]",
                    metadata={"module": ref.replace("/", ".").rsplit(".", 1)[0]},
                )
            )
        elif drift_kind == "contract":
            actions.append(
                DriftAction(
                    drift_type="contract",
                    stale_ref=ref,
                    current_ref=ref + " /*updated*/",
                    metadata={"function": ref.split("(")[0], "old_params": [], "new_params": []},
                )
            )
        else:
            actions.append(
                DriftAction(
                    drift_type="rename",
                    stale_ref=ref,
                    current_ref=ref + "_v2",
                    metadata={},
                )
            )

    # We synthesize a drifted codebase consistent with these actions so the
    # env's invariant validator does not reject the inject.
    base = build_base_codebase()
    drifted = copy.deepcopy(base)
    for a in actions:
        if a.drift_type == "rename":
            drifted.functions[a.stale_ref + "_real_pr_marker"] = "args"
            drifted.functions[a.current_ref] = "args"
        elif a.drift_type == "removal":
            drifted.files = [f for f in drifted.files if f != a.stale_ref]
        elif a.drift_type == "contract":
            fn = a.metadata.get("function", a.stale_ref.split("(")[0])
            drifted.api_signatures[fn] = []  # neutral; new_params == [] satisfies invariant

    env = CodeDriftEnv()
    env.inject_episode(
        drifted=drifted,
        actions=actions,
        pr_diff=pr_diff,
        base=base,
        validate=False,  # real PRs may not contain catalog-style stale tokens verbatim
    )
    obs, reward, done, info = env.step(review)
    summary = detect_languages(pr_diff).__dict__
    return float(reward), info, summary
