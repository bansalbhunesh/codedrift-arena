"""Reviewer parser tolerance tests."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents_v2.reviewer_io import (
    MalformedPrediction,
    ReviewerPrediction,
    parse_reviewer_output,
)


class TestReviewerIO(unittest.TestCase):
    def test_clean_json_object(self) -> None:
        out = parse_reviewer_output(
            json.dumps(
                {
                    "verdict": "REQUEST_CHANGES",
                    "root_cause": "src/orders.py::createOrder",
                    "failure_path": ["a", "b", "c"],
                    "confidence": 0.85,
                    "reasoning": "x",
                }
            )
        )
        self.assertIsInstance(out, ReviewerPrediction)
        self.assertEqual(out.verdict, "REQUEST_CHANGES")
        self.assertEqual(out.root_cause, "src/orders.py::createOrder")
        self.assertAlmostEqual(out.confidence, 0.85)
        self.assertEqual(out.failure_path, ["a", "b", "c"])

    def test_fenced_json(self) -> None:
        text = '```json\n{"verdict": "APPROVE", "root_cause": "", "failure_path": [], "confidence": 0.6}\n```'
        out = parse_reviewer_output(text)
        self.assertIsInstance(out, ReviewerPrediction)
        self.assertEqual(out.verdict, "APPROVE")

    def test_key_value_legacy(self) -> None:
        out = parse_reviewer_output(
            "VERDICT: REQUEST_CHANGES\nROOT_CAUSE: src/orders.py::createOrder\nFAILURE_PATH: a -> b -> c\nCONFIDENCE: 0.7\n"
        )
        self.assertIsInstance(out, ReviewerPrediction)
        self.assertEqual(out.verdict, "REQUEST_CHANGES")
        self.assertEqual(out.failure_path, ["a", "b", "c"])

    def test_invalid_verdict_marked_malformed(self) -> None:
        out = parse_reviewer_output(json.dumps({"verdict": "MAYBE", "root_cause": ""}))
        self.assertIsInstance(out, MalformedPrediction)

    def test_empty_output(self) -> None:
        out = parse_reviewer_output("")
        self.assertIsInstance(out, MalformedPrediction)

    def test_confidence_clipped(self) -> None:
        out = parse_reviewer_output(json.dumps({"verdict": "APPROVE", "confidence": 5.0}))
        self.assertIsInstance(out, ReviewerPrediction)
        self.assertEqual(out.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
