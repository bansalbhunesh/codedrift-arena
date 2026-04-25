"""Generator agent: every pattern produces a valid AST and a real failure."""

from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents_v2.generator_agent import ALL_PATTERNS, GeneratorAgent
from env_v2.base_repo_loader import copy_base_repo, list_source_files
from env_v2.exec_engine import run_pytest


class TestGenerator(unittest.TestCase):
    def test_every_pattern_emits_parseable_python_and_failure(self) -> None:
        for pat in ALL_PATTERNS:
            with self.subTest(pattern=pat):
                td = Path(tempfile.mkdtemp(prefix=f"v2_gen_{pat}_"))
                repo = copy_base_repo(td / "repo")
                outcome = GeneratorAgent(seed=21).generate(
                    repo, difficulty="easy", forced_patterns=[pat]
                )
                self.assertEqual(len(outcome.mutations), 1)
                # Source must still parse.
                for src in list_source_files(repo):
                    ast.parse(src.read_text(encoding="utf-8"))
                # Every pattern must visibly break the suite.
                res = run_pytest(repo, timeout_s=20)
                self.assertGreaterEqual(res.failed + res.errors, 1)
                self.assertTrue(outcome.pr_diff)

    def test_difficulty_caps_mutation_count(self) -> None:
        for difficulty, expected in [("easy", 1), ("medium", 2), ("hard", 3)]:
            td = Path(tempfile.mkdtemp(prefix=f"v2_gen_diff_{difficulty}_"))
            repo = copy_base_repo(td / "repo")
            outcome = GeneratorAgent(seed=33).generate(repo, difficulty=difficulty)
            self.assertLessEqual(len(outcome.mutations), expected)


if __name__ == "__main__":
    unittest.main()
