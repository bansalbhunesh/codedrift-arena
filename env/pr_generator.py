"""
PRDiffGenerator — generates realistic PR diffs that embed stale references.

Key design decision: diffs must contain BOTH clean code and stale code.
If every line is stale, the agent can just reject everything and win.
The challenge is spotting the one bad line in otherwise reasonable code.
"""

import random

from agents.drift_agent import DriftAction

DIFF_FILENAMES = [
    "src/feature.py",
    "src/checkout_flow.py",
    "services/api_handler.py",
    "lib/sync_worker.py",
]

# Clean code snippets that are always valid — pad diffs to look realistic
CLEAN_SNIPPETS = [
    ("+from models.user import User", "import"),
    ("+from models.order import Order", "import"),
    ("+from config.settings import Config", "import"),
    ("+from utils.helpers import format_response", "import"),
    ("+", "blank"),
    ("+logger = logging.getLogger(__name__)", "setup"),
    ("+config = Config.load()", "setup"),
    ("+user = User.get(user_id)", "logic"),
    ("+order = Order.find(order_id)", "logic"),
    ("+if not user:", "logic"),
    ("+    raise ValueError('User not found')", "logic"),
    ("+response = format_response(data)", "logic"),
    ("+return {'status': 'ok', 'data': response}", "return"),
]


class PRDiffGenerator:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def generate(self, actions: list[DriftAction], filename: str | None = None) -> str:
        """
        Generates a unified diff that:
        - Looks like a real feature PR
        - Embeds one stale reference per DriftAction
        - Surrounds stale refs with plausible clean code
        """
        fn = filename or self.rng.choice(DIFF_FILENAMES)
        lines = [
            f"diff --git a/{fn} b/{fn}",
            f"--- a/{fn}",
            f"+++ b/{fn}",
            "@@ -1,15 +1,30 @@",
        ]

        clean_imports = [s for s, t in CLEAN_SNIPPETS if t == "import"]
        for snippet in self.rng.sample(clean_imports, k=min(3, len(clean_imports))):
            lines.append(snippet)

        lines.append("+")
        lines.append("+def process_feature(user_id: str, **kwargs):")

        clean_logic = [s for s, t in CLEAN_SNIPPETS if t == "logic"]
        stale_lines = self._build_stale_lines(actions)

        body_lines = clean_logic[:2] + stale_lines + clean_logic[2:4]
        self.rng.shuffle(body_lines)
        for line in body_lines:
            inner = line[1:] if line.startswith("+") else line
            inner = inner.lstrip()
            lines.append(f"+    {inner}")

        lines.append("+    return {'status': 'ok'}")

        return "\n".join(lines)

    def _build_stale_lines(self, actions: list[DriftAction]) -> list[str]:
        stale_lines = []
        for action in actions:
            pat = action.bug_pattern or action.drift_type
            if pat == "partial_rename":
                stale_lines.extend(self._partial_rename_lines(action))
            elif pat == "null_missing":
                stale_lines.extend(self._null_missing_lines(action))
            elif pat == "type_mismatch":
                stale_lines.extend(self._type_mismatch_lines(action))
            elif pat == "condition_flip":
                stale_lines.extend(self._condition_flip_lines(action))
            elif pat == "off_by_one":
                stale_lines.extend(self._off_by_one_lines(action))
            elif action.drift_type == "rename":
                stale_lines.append(f"data = {action.stale_ref}(user_id)")
            elif action.drift_type == "removal":
                module = action.metadata.get("module", action.stale_ref)
                stale_lines.append(f"from {module} import helper  # stale import")
            elif action.drift_type == "contract":
                fn = action.metadata.get("function", "unknown")
                old_params = action.metadata.get("old_params", [])
                param_str = ", ".join(old_params)
                stale_lines.append(f"result = {fn}({param_str})")
        return stale_lines

    def _partial_rename_lines(self, action: DriftAction) -> list[str]:
        """Mixed diff: fresh_context uses new name correctly, stale_context doesn't."""
        old = action.stale_ref
        new = action.current_ref
        fresh_ctx = action.metadata.get("fresh_context", "update_flow")
        stale_ctx = action.metadata.get("stale_context", "legacy_path")
        return [
            f"# {fresh_ctx}: correctly updated",
            f"profile = {new}(user_id)  # updated",
            f"# {stale_ctx}: missed this one",
            f"cached = {old}(user_id)   # BUG: stale reference",
        ]

    def _null_missing_lines(self, action: DriftAction) -> list[str]:
        """PR accesses attribute on result of function that now returns Optional."""
        fn = action.metadata.get("function", "get_data")
        attr = action.metadata.get("nullable_attribute", "value")
        return [
            f"result = {fn}(user_id)",
            f"value = result.{attr}  # BUG: result may be None",
        ]

    def _type_mismatch_lines(self, action: DriftAction) -> list[str]:
        """PR passes old type (int) where new type (str) is expected."""
        fn = action.metadata.get("function", "process")
        param = action.metadata.get("param", "id")
        old_example = action.metadata.get("old_example", "123")
        return [
            f"order = {fn}({param}={old_example})  # BUG: {param} should now be a string",
        ]

    def _condition_flip_lines(self, action: DriftAction) -> list[str]:
        """PR passes old boolean value whose meaning has been inverted."""
        fn = action.metadata.get("function", "validate")
        param = action.metadata.get("param", "strict")
        old_value = action.metadata.get("old_value", "True")
        new_semantics = action.metadata.get("new_semantics", "semantics changed")
        return [
            f"result = {fn}(data, {param}={old_value})  # BUG: {param} semantics inverted",
            f"# Note: {new_semantics}",
        ]

    def _off_by_one_lines(self, action: DriftAction) -> list[str]:
        """PR uses 1-based index where 0-based is now expected."""
        old_call = action.metadata.get("old_call", "getPage(page=1)")
        new_convention = action.metadata.get("new_convention", "0-based")
        return [
            f"items = {old_call}  # BUG: now uses {new_convention}",
        ]
