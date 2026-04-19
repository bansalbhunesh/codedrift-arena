"""
PRDiffGenerator — generates realistic PR diffs that embed stale references.

Key design decision: diffs must contain BOTH clean code and stale code.
If every line is stale, the agent can just reject everything and win.
The challenge is spotting the one bad line in otherwise reasonable code.
"""

import random

from agents.drift_agent import DriftAction

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
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)

    def generate(self, actions: list[DriftAction], filename: str = "src/feature.py") -> str:
        """
        Generates a unified diff that:
        - Looks like a real feature PR
        - Embeds one stale reference per DriftAction
        - Surrounds stale refs with plausible clean code
        """
        lines = [
            f"diff --git a/{filename} b/{filename}",
            f"--- a/{filename}",
            f"+++ b/{filename}",
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
            if action.drift_type == "rename":
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
