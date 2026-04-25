"""Causal scorer behavior: every component contributes to the right direction."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents_v2.reviewer_io import MalformedPrediction, parse_reviewer_output
from env_v2.exec_engine import ExecutionResult, FailedTest
from rewards_v2.causal_scorer import CausalScorer, MALFORMED_PENALTY


def _exec_with_one_failure() -> ExecutionResult:
    return ExecutionResult(
        returncode=1,
        duration_s=0.1,
        passed=5,
        failed=1,
        errors=0,
        failed_tests=[
            FailedTest(
                nodeid="tests/test_orders.py::test_create_order_two_args",
                file="tests/test_orders.py",
                line=10,
                exception="TypeError",
                message="missing required argument",
                traceback="...",
                call_chain=["src/orders.py:5"],
            )
        ],
        stdout_tail="",
        stderr_tail="",
    )


GROUND_TRUTH = {
    "verdict": "REQUEST_CHANGES",
    "root_cause": "src/orders.py::createOrder",
    "root_cause_symbols": ["src/orders.py::createOrder"],
    "failure_path": [
        "tests/test_orders.py::test_create_order_two_args",
        "createOrder",
        "src/orders.py::createOrder",
    ],
    "failing_tests": ["tests/test_orders.py::test_create_order_two_args"],
}


class TestCausalScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = CausalScorer()
        self.exec_result = _exec_with_one_failure()

    def test_perfect_prediction_high_reward(self) -> None:
        pred = parse_reviewer_output(
            json.dumps(
                {
                    "verdict": "REQUEST_CHANGES",
                    "root_cause": "src/orders.py::createOrder",
                    "failure_path": GROUND_TRUTH["failure_path"],
                    "confidence": 0.9,
                }
            )
        )
        reward, info = self.scorer.score(pred, GROUND_TRUTH, self.exec_result, mutations=[])
        self.assertGreater(reward, 1.5)
        self.assertEqual(info["episode_outcome"], "perfect")

    def test_malformed_gets_fixed_penalty(self) -> None:
        pred = parse_reviewer_output("garbage not json")
        reward, info = self.scorer.score(pred, GROUND_TRUTH, self.exec_result, mutations=[])
        self.assertEqual(reward, MALFORMED_PENALTY)
        self.assertEqual(info["episode_outcome"], "malformed")

    def test_hallucination_penalized(self) -> None:
        pred = parse_reviewer_output(
            json.dumps(
                {
                    "verdict": "REQUEST_CHANGES",
                    "root_cause": "fake/file.py::nonexistent_symbol",
                    "failure_path": [],
                    "confidence": 0.95,
                }
            )
        )
        reward, info = self.scorer.score(pred, GROUND_TRUTH, self.exec_result, mutations=[])
        self.assertLess(reward, 0.0)
        self.assertGreater(info["breakdown"]["hallucination"], 0.0)
        self.assertIn("nonexistent_symbol", info["hallucinated_tokens"])

    def test_partial_credit_symbol_only(self) -> None:
        pred = parse_reviewer_output(
            json.dumps(
                {
                    "verdict": "REQUEST_CHANGES",
                    "root_cause": "createOrder",
                    "failure_path": [],
                    "confidence": 0.5,
                }
            )
        )
        reward, info = self.scorer.score(pred, GROUND_TRUTH, self.exec_result, mutations=[])
        self.assertGreater(info["breakdown"]["root_cause"], 0.5)
        self.assertLess(info["breakdown"]["root_cause"], 1.0)


if __name__ == "__main__":
    unittest.main()
