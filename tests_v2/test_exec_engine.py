"""Exec engine sanity tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents_v2.generator_agent import GeneratorAgent
from env_v2.base_repo_loader import copy_base_repo
from env_v2.exec_engine import run_pytest


class TestExecEngine(unittest.TestCase):
    def test_clean_repo_passes(self) -> None:
        td = Path(tempfile.mkdtemp(prefix="v2_exec_clean_"))
        repo = copy_base_repo(td / "repo")
        result = run_pytest(repo, timeout_s=20)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.errors, 0)
        self.assertGreaterEqual(result.passed, 1)

    def test_contract_mutation_breaks_a_test(self) -> None:
        td = Path(tempfile.mkdtemp(prefix="v2_exec_contract_"))
        repo = copy_base_repo(td / "repo")
        GeneratorAgent(seed=11).generate(repo, difficulty="easy", forced_patterns=["contract"])
        result = run_pytest(repo, timeout_s=20)
        self.assertGreaterEqual(result.failed + result.errors, 1)
        self.assertGreaterEqual(len(result.failed_tests), 1)
        self.assertTrue(result.failed_tests[0].nodeid)

    def test_collection_error_captured(self) -> None:
        td = Path(tempfile.mkdtemp(prefix="v2_exec_rename_"))
        repo = copy_base_repo(td / "repo")
        GeneratorAgent(seed=12).generate(repo, difficulty="easy", forced_patterns=["rename"])
        result = run_pytest(repo, timeout_s=20)
        # rename causes ImportError at collection time.
        self.assertGreaterEqual(result.errors, 1)
        self.assertGreaterEqual(len(result.failed_tests), 1)


if __name__ == "__main__":
    unittest.main()
