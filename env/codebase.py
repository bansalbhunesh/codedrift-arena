"""
Codebase state — shared data model used by env, drift agent, and reward scorer.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CodebaseState:
    """
    Snapshot of a codebase at a point in time.
    The drift agent mutates this. The reviewer sees the mutated version.
    The PR diff contains stale references to the PRE-mutation state.
    """

    functions: Dict[str, str] = field(default_factory=dict)  # name -> signature string
    files: List[str] = field(default_factory=list)  # available file paths
    api_signatures: Dict[str, List[str]] = field(default_factory=dict)  # fn name -> param list
    version: int = 0  # bumped on each drift

    def clone(self) -> "CodebaseState":
        return copy.deepcopy(self)

    def summary(self) -> str:
        lines = [f"Codebase v{self.version}"]
        lines.append(f"  {len(self.functions)} functions")
        lines.append(f"  {len(self.files)} files")
        lines.append(f"  {len(self.api_signatures)} tracked API signatures")
        return "\n".join(lines)


def build_base_codebase() -> CodebaseState:
    """
    Canonical starting codebase for every episode.
    Rich enough to support all drift types across multiple episodes.
    """
    state = CodebaseState()
    state.functions = {
        "getUserData": "userId: str",
        "createOrder": "item: str, qty: int",
        "deleteRecord": "recordId: str",
        "sendNotification": "userId: str, message: str",
        "validateInput": "data: dict",
        "parseResponse": "response: dict",
        "authenticate": "username: str, password: str",
        "sendEmail": "to: str, subject: str, body: str",
        "loadConfig": "env: str",
        "checkPermission": "userId: str, resource: str",
        "logEvent": "event: str, metadata: dict",
        "refreshToken": "token: str",
    }
    state.files = [
        "utils/helpers.py",
        "utils/legacy.py",
        "services/auth.py",
        "services/v1_client.py",
        "lib/old_auth.py",
        "helpers/deprecated.py",
        "adapters/xml_parser.py",
        "connectors/soap_bridge.py",
        "models/user.py",
        "models/order.py",
        "models/product.py",
        "config/settings.py",
    ]
    state.api_signatures = {
        "createOrder": ["item", "qty"],
        "sendEmail": ["to", "subject", "body"],
        "authenticate": ["username", "password"],
        "fetchUserData": ["userId"],
        "submitOrder": ["item", "qty", "userId"],
    }
    return state
