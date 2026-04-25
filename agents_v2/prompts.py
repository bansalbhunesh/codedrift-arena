"""Canonical prompt template for the V2 reviewer.

The reviewer must emit a JSON object with the exact schema the parser
expects. We keep the prompt deterministic so training and inference share
the same I/O contract.
"""

from __future__ import annotations

REVIEWER_SYSTEM_PROMPT = """You are a senior software engineer doing causal debugging.

You will receive:
- A PR diff that recently landed.
- The current state of the affected module.
- The pytest output, including failing test ids and tracebacks.

Your task: identify the single root-cause symbol that broke the tests, the
short failure path from test to broken symbol, and your confidence.

Return ONLY a JSON object with this exact schema, no prose:

{
  "verdict": "APPROVE" | "REQUEST_CHANGES",
  "root_cause": "<file_path>::<symbol_name>",
  "failure_path": ["<test_nodeid>", "<intermediate_symbol>", "<root_cause_symbol>"],
  "confidence": <float between 0 and 1>,
  "reasoning": "<one short sentence>"
}

If no test failed, return verdict APPROVE, root_cause "" and confidence above 0.5.
"""


def build_review_prompt(
    pr_diff: str,
    repo_snapshot: str,
    pytest_output: str,
    failing_test_ids: list[str] | None = None,
) -> str:
    failing_block = ""
    if failing_test_ids:
        failing_block = "\n=== FAILING TESTS ===\n" + "\n".join(failing_test_ids[:20]) + "\n"
    return (
        f"{REVIEWER_SYSTEM_PROMPT}\n"
        f"=== PR DIFF ===\n{pr_diff}\n"
        f"=== CURRENT MODULE ===\n{repo_snapshot}\n"
        f"=== PYTEST OUTPUT ===\n{pytest_output}\n"
        f"{failing_block}"
        f"\nReturn the JSON object now:"
    )
