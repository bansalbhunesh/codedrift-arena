"""V2 generator — applies real bug-pattern mutations to the base_repo source.

Strategy:
- Each mutation is implemented with the stdlib :mod:`ast` module so we never
  produce syntax-broken code; ``ast.parse`` validates the rewritten source.
- Mutations target ``src/orders.py`` only (the v2 base repo). Tests are
  immutable so they remain a faithful oracle.
- Each call returns a list of :class:`MutationResult` describing what was
  changed; this is the ground truth the reward scorer compares against.

Difficulty:
- ``easy``   -> 1 mutation
- ``medium`` -> 2 mutations
- ``hard``   -> 3 mutations
Mutations are sampled without replacement to avoid colliding rewrites.
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from env_v2.base_repo_loader import list_source_files

ALL_PATTERNS: list[str] = [
    "rename",
    "removal",
    "contract",
    "partial_rename",
    "null_missing",
    "type_mismatch",
    "condition_flip",
    "off_by_one",
    "cascade",
]

DIFFICULTY_TO_COUNT: dict[str, int] = {"easy": 1, "medium": 2, "hard": 3}


@dataclass
class MutationResult:
    pattern: str
    file: str
    symbol: str
    detail: dict = field(default_factory=dict)

    @property
    def root_cause(self) -> str:
        return f"{self.file}::{self.symbol}"


@dataclass
class GenerationOutcome:
    mutations: list[MutationResult]
    pr_diff: str
    seed: int
    requested: list[str]


class _RenameFunctionTransformer(ast.NodeTransformer):
    def __init__(self, old: str, new: str):
        self.old = old
        self.new = new
        self.changed = False

    def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
        if node.name == self.old:
            node.name = self.new
            self.changed = True
        return self.generic_visit(node)


class _RemoveFunctionTransformer(ast.NodeTransformer):
    def __init__(self, name: str):
        self.name = name
        self.removed = False

    def visit_Module(self, node: ast.Module):  # type: ignore[override]
        new_body = []
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == self.name:
                self.removed = True
                continue
            new_body.append(child)
        node.body = new_body
        return node


class _AddRequiredArgTransformer(ast.NodeTransformer):
    def __init__(self, fn_name: str, new_param: str):
        self.fn_name = fn_name
        self.new_param = new_param
        self.changed = False

    def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
        if node.name != self.fn_name:
            return node
        existing = {a.arg for a in node.args.args}
        if self.new_param in existing:
            return node
        node.args.args.append(ast.arg(arg=self.new_param, annotation=None))
        # Touch the body so the new param is "used" — prevents lint noise and
        # makes a meaningful runtime difference if callers don't supply it.
        guard = ast.parse(
            f"if {self.new_param} is None:\n    raise TypeError({self.new_param!r} + ' is required')\n"
        ).body
        node.body = list(guard) + node.body
        self.changed = True
        return node


class _NullableReturnTransformer(ast.NodeTransformer):
    """Force a function to return ``None`` instead of its real value."""

    def __init__(self, fn_name: str):
        self.fn_name = fn_name
        self.changed = False

    def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
        if node.name != self.fn_name:
            return node
        node.body = [ast.Return(value=ast.Constant(value=None))]
        self.changed = True
        return node


class _IntToStrCastTransformer(ast.NodeTransformer):
    """Replace ``int(x)`` -> ``str(x)`` inside a target function body."""

    def __init__(self, fn_name: str):
        self.fn_name = fn_name
        self.changed = False
        self._inside = False

    def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
        if node.name != self.fn_name:
            return node
        self._inside = True
        self.generic_visit(node)
        self._inside = False
        return node

    def visit_Call(self, node: ast.Call):  # type: ignore[override]
        if self._inside and isinstance(node.func, ast.Name) and node.func.id == "int":
            node.func = ast.Name(id="str", ctx=ast.Load())
            self.changed = True
        return self.generic_visit(node)


class _BoolFlipTransformer(ast.NodeTransformer):
    """Flip ``if strict:`` to ``if not strict:`` inside a target function."""

    def __init__(self, fn_name: str, var: str):
        self.fn_name = fn_name
        self.var = var
        self.changed = False
        self._inside = False

    def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
        if node.name != self.fn_name:
            return node
        self._inside = True
        self.generic_visit(node)
        self._inside = False
        return node

    def visit_If(self, node: ast.If):  # type: ignore[override]
        if (
            self._inside
            and isinstance(node.test, ast.Name)
            and node.test.id == self.var
        ):
            node.test = ast.UnaryOp(op=ast.Not(), operand=ast.Name(id=self.var, ctx=ast.Load()))
            self.changed = True
        return self.generic_visit(node)


class _OffByOneTransformer(ast.NodeTransformer):
    """Subtract 1 from any ``page - 1`` style expression inside ``fn_name``."""

    def __init__(self, fn_name: str, var: str = "page"):
        self.fn_name = fn_name
        self.var = var
        self.changed = False
        self._inside = False

    def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
        if node.name != self.fn_name:
            return node
        self._inside = True
        self.generic_visit(node)
        self._inside = False
        return node

    def visit_BinOp(self, node: ast.BinOp):  # type: ignore[override]
        if (
            self._inside
            and isinstance(node.op, ast.Sub)
            and isinstance(node.left, ast.Name)
            and node.left.id == self.var
            and isinstance(node.right, ast.Constant)
            and node.right.value == 1
        ):
            # Off-by-one bug: subtract 2 instead of 1.
            node.right = ast.Constant(value=2)
            self.changed = True
        return self.generic_visit(node)


def _apply_transformer(source: str, transformer: ast.NodeTransformer) -> tuple[str, bool]:
    tree = ast.parse(source)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    rewritten = ast.unparse(new_tree)
    # Validate rewritten source still parses.
    ast.parse(rewritten)
    return rewritten, getattr(transformer, "changed", getattr(transformer, "removed", False))


# ── Pattern definitions: each mutator picks a target function from orders.py.

@dataclass
class _PatternSpec:
    name: str
    target_fn: str
    builder: Callable[[], ast.NodeTransformer]
    new_symbol_name: Optional[str] = None
    detail: dict = field(default_factory=dict)


def _spec_rename() -> _PatternSpec:
    return _PatternSpec(
        name="rename",
        target_fn="fetchUserData",
        builder=lambda: _RenameFunctionTransformer("fetchUserData", "lookupUser"),
        new_symbol_name="lookupUser",
        detail={"from": "fetchUserData", "to": "lookupUser"},
    )


def _spec_removal() -> _PatternSpec:
    return _PatternSpec(
        name="removal",
        target_fn="sendNotification",
        builder=lambda: _RemoveFunctionTransformer("sendNotification"),
        detail={"removed": "sendNotification"},
    )


def _spec_contract() -> _PatternSpec:
    return _PatternSpec(
        name="contract",
        target_fn="createOrder",
        builder=lambda: _AddRequiredArgTransformer("createOrder", "userId"),
        detail={"added_param": "userId"},
    )


def _spec_partial_rename() -> _PatternSpec:
    # partial_rename: rename happens in source but tests still call old name
    # (matches the v1 semantic; tests will fail on AttributeError).
    return _PatternSpec(
        name="partial_rename",
        target_fn="validateInput",
        builder=lambda: _RenameFunctionTransformer("validateInput", "sanitizeInput"),
        new_symbol_name="sanitizeInput",
        detail={"from": "validateInput", "to": "sanitizeInput"},
    )


def _spec_null_missing() -> _PatternSpec:
    return _PatternSpec(
        name="null_missing",
        target_fn="fetchUserData",
        builder=lambda: _NullableReturnTransformer("fetchUserData"),
        detail={"function_returns_none": "fetchUserData"},
    )


def _spec_type_mismatch() -> _PatternSpec:
    return _PatternSpec(
        name="type_mismatch",
        target_fn="sendNotification",
        builder=lambda: _IntToStrCastTransformer("sendNotification"),
        detail={"function": "sendNotification", "param_cast": "int->str"},
    )


def _spec_condition_flip() -> _PatternSpec:
    return _PatternSpec(
        name="condition_flip",
        target_fn="validateInput",
        builder=lambda: _BoolFlipTransformer("validateInput", "strict"),
        detail={"function": "validateInput", "flipped": "strict"},
    )


def _spec_off_by_one() -> _PatternSpec:
    return _PatternSpec(
        name="off_by_one",
        target_fn="getPageItems",
        builder=lambda: _OffByOneTransformer("getPageItems", "page"),
        detail={"function": "getPageItems", "var": "page"},
    )


def _spec_cascade() -> _PatternSpec:
    """Hidden cause: break the deepest function in a 3-frame chain.

    The base repo defines ``process_order -> enrich_user -> fetchUserData``.
    We mutate ``fetchUserData`` to return ``None`` so the test failure surfaces
    inside ``enrich_user`` (KeyError on ``user["id"]``) — the agent must trace
    two frames back to identify ``fetchUserData`` as the actual root cause.
    """
    return _PatternSpec(
        name="cascade",
        target_fn="fetchUserData",
        builder=lambda: _NullableReturnTransformer("fetchUserData"),
        detail={
            "function": "fetchUserData",
            "chain_depth": 3,
            "surface_function": "process_order",
            "intermediate": "enrich_user",
            "hidden_cause": "fetchUserData returns None; enrich_user crashes; process_order fails",
        },
    )


_PATTERN_BUILDERS: dict[str, Callable[[], _PatternSpec]] = {
    "rename": _spec_rename,
    "removal": _spec_removal,
    "contract": _spec_contract,
    "partial_rename": _spec_partial_rename,
    "null_missing": _spec_null_missing,
    "type_mismatch": _spec_type_mismatch,
    "condition_flip": _spec_condition_flip,
    "off_by_one": _spec_off_by_one,
    "cascade": _spec_cascade,
}


class GeneratorAgent:
    """Mutates files inside a copied base_repo and returns ground-truth labels."""

    def __init__(self, seed: Optional[int] = None, allowed_patterns: Optional[list[str]] = None):
        self.rng = random.Random(seed)
        self.allowed = list(allowed_patterns) if allowed_patterns else list(ALL_PATTERNS)

    def generate(
        self,
        repo_root: Path,
        difficulty: str = "easy",
        forced_patterns: Optional[list[str]] = None,
    ) -> GenerationOutcome:
        """Mutate ``repo_root`` in place and return :class:`GenerationOutcome`."""
        if difficulty not in DIFFICULTY_TO_COUNT:
            raise ValueError(f"unknown difficulty {difficulty!r}")
        n = DIFFICULTY_TO_COUNT[difficulty]
        pool = forced_patterns if forced_patterns else self.allowed
        if not pool:
            raise ValueError("no patterns available to sample from")
        chosen = self.rng.sample(pool, k=min(n, len(pool)))

        sources = list_source_files(repo_root)
        target_path = next((p for p in sources if p.name == "orders.py"), None)
        if target_path is None:
            raise FileNotFoundError("base_repo/src/orders.py missing")
        original_source = target_path.read_text(encoding="utf-8")
        source = original_source

        results: list[MutationResult] = []
        applied_targets: set[str] = set()
        for name in chosen:
            spec = _PATTERN_BUILDERS[name]()
            # If the target function was already renamed/removed, skip to keep
            # mutations independent. The reward function still sees the
            # remaining mutations as ground truth.
            if spec.target_fn in applied_targets:
                continue
            transformer = spec.builder()
            try:
                rewritten, changed = _apply_transformer(source, transformer)
            except SyntaxError:
                continue
            if not changed:
                continue
            source = rewritten
            applied_targets.add(spec.target_fn)
            symbol = spec.new_symbol_name or spec.target_fn
            results.append(
                MutationResult(
                    pattern=spec.name,
                    file=str(target_path.relative_to(repo_root)).replace("\\", "/"),
                    symbol=symbol,
                    detail={**spec.detail, "original_target": spec.target_fn},
                )
            )

        target_path.write_text(source, encoding="utf-8")
        pr_diff = _build_inline_diff(
            file=str(target_path.relative_to(repo_root)).replace("\\", "/"),
            before=original_source,
            after=source,
        )
        return GenerationOutcome(
            mutations=results,
            pr_diff=pr_diff,
            seed=self.rng.randint(0, 2**31 - 1),
            requested=chosen,
        )


def _build_inline_diff(file: str, before: str, after: str) -> str:
    """Tiny unified diff so the agent receives a PR-style observation."""
    import difflib

    lines = list(
        difflib.unified_diff(
            before.splitlines(keepends=False),
            after.splitlines(keepends=False),
            fromfile=f"a/{file}",
            tofile=f"b/{file}",
            lineterm="",
            n=3,
        )
    )
    return "\n".join(lines) if lines else f"# no-op diff for {file}\n"
