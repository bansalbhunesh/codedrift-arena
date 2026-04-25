"""
FailureCascadeSimulator — converts DriftActions into realistic test failure output.

This is the "execution engine" that makes causal reasoning mandatory.
Instead of: "find the stale ref in the diff"
Now it's:   "trace failing tests → call chain → root mutation"

Judges see an observable execution loop:
  env mutates codebase → tests fail → reviewer traces cascade → reward on accuracy
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from agents.drift_agent import DriftAction

# ── Lookup tables: stale identifier → (test_suite, test_name, calling_fn) ────

_RENAME_TESTS: dict[str, tuple[str, str, str]] = {
    "getUserData":        ("test_user_management",  "test_get_user_by_id",        "process_feature"),
    "fetchUserData":      ("test_user_management",  "test_get_user_by_id",        "process_feature"),
    "createOrder":        ("test_orders",            "test_create_basic_order",    "checkout_flow"),
    "submitOrder":        ("test_orders",            "test_submit_order",          "checkout_flow"),
    "deleteRecord":       ("test_records",           "test_delete_record",         "record_manager"),
    "removeRecord":       ("test_records",           "test_delete_record",         "record_manager"),
    "sendNotification":   ("test_notifications",     "test_send_alert",            "notification_service"),
    "dispatchAlert":      ("test_notifications",     "test_send_alert",            "notification_service"),
    "validateInput":      ("test_validation",        "test_validate_form_input",   "form_handler"),
    "sanitizeInput":      ("test_validation",        "test_validate_form_input",   "form_handler"),
    "parseResponse":      ("test_api",               "test_parse_api_response",    "api_client"),
    "deserializeResponse":("test_api",               "test_parse_api_response",    "api_client"),
    "loadConfig":         ("test_config",            "test_load_app_config",       "app_init"),
    "readConfig":         ("test_config",            "test_load_app_config",       "app_init"),
    "checkPermission":    ("test_auth",              "test_check_user_permission", "auth_middleware"),
    "verifyAccess":       ("test_auth",              "test_check_user_permission", "auth_middleware"),
}

_REMOVAL_TESTS: dict[str, tuple[str, str, str, str]] = {
    "utils/legacy.py":        ("test_compatibility", "test_legacy_helper_import",  "init_services",    "utils.legacy"),
    "helpers/deprecated.py":  ("test_compatibility", "test_deprecated_helpers",    "migration_runner", "helpers.deprecated"),
    "lib/old_auth.py":        ("test_auth",          "test_legacy_auth_flow",      "auth_bootstrap",   "lib.old_auth"),
    "services/v1_client.py":  ("test_api",           "test_v1_client_init",        "client_factory",   "services.v1_client"),
    "adapters/xml_parser.py": ("test_parsers",       "test_xml_parse_response",    "response_parser",  "adapters.xml_parser"),
    "connectors/soap_bridge.py": ("test_connectors", "test_soap_bridge_connect",   "connector_pool",   "connectors.soap_bridge"),
}

_CONTRACT_TESTS: dict[str, tuple[str, str, str]] = {
    "createOrder":   ("test_orders",        "test_create_order_basic",     "order_service"),
    "sendEmail":     ("test_notifications", "test_send_welcome_email",     "email_dispatcher"),
    "authenticate":  ("test_auth",          "test_user_login",             "auth_controller"),
    "fetchUserData": ("test_users",         "test_fetch_user_profile",     "user_service"),
    "submitOrder":   ("test_orders",        "test_submit_warehouse_order", "warehouse_client"),
}

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class TestFailure:
    test_suite: str
    test_name: str
    error_type: str
    error_message: str
    # Chain from test entry-point down to the broken call (inner-most last).
    # This is the ground truth for FAILURE_PATH scoring.
    call_chain: list[str]
    downstream_blocked: list[str] = field(default_factory=list)

    def path_tokens(self) -> list[str]:
        """Unique tokens that a correct FAILURE_PATH answer must contain."""
        return list(dict.fromkeys(self.call_chain + [self.test_name]))


@dataclass
class FailureCascade:
    failures: list[TestFailure]
    total_failing: int
    total_blocked: int

    def format_for_prompt(self) -> str:
        if not self.failures:
            return "=== TEST EXECUTION OUTPUT ===\nAll tests passed. ✓\n"

        lines = ["=== TEST EXECUTION OUTPUT ==="]
        lines.append(
            f"Running tests... FAILED "
            f"({self.total_failing} error(s), {self.total_blocked} blocked)\n"
        )
        for f in self.failures:
            lines.append(f"FAIL {f.test_suite}::{f.test_name}")
            lines.append(f"  {f.error_type}: {f.error_message}")
            chain_str = " -> ".join(f.call_chain)
            lines.append(f"  Call chain: {chain_str}")
            if f.downstream_blocked:
                blocked = ", ".join(f.downstream_blocked)
                lines.append(f"  Blocked: {blocked}")
            lines.append("")
        return "\n".join(lines)

    def all_path_tokens(self) -> list[str]:
        """All tokens across all failures — used by scorer for FAILURE_PATH grading."""
        seen: dict[str, None] = {}
        for f in self.failures:
            for tok in f.path_tokens():
                seen[tok] = None
        return list(seen)


# ── Simulator ─────────────────────────────────────────────────────────────────


class FailureCascadeSimulator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def simulate(self, actions: list[DriftAction]) -> FailureCascade:
        if not actions:
            return FailureCascade(failures=[], total_failing=0, total_blocked=0)

        failures: list[TestFailure] = []
        total_blocked = 0

        for action in actions:
            failure = self._make_failure(action)
            if failure is None:
                continue
            n_blocked = self.rng.randint(0, 2)
            if n_blocked:
                stale_slug = action.stale_ref.split("(")[0].lower().replace("/", "_").replace(".", "_")
                failure.downstream_blocked = [
                    f"test_{stale_slug}_{i}" for i in range(n_blocked)
                ]
                total_blocked += n_blocked
            failures.append(failure)

        return FailureCascade(
            failures=failures,
            total_failing=len(failures),
            total_blocked=total_blocked,
        )

    # ── Failure factories ─────────────────────────────────────────────────────

    def _make_failure(self, action: DriftAction) -> Optional[TestFailure]:
        pat = action.bug_pattern or action.drift_type
        if pat == "partial_rename":
            return self._partial_rename_failure(action)
        if pat == "null_missing":
            return self._null_missing_failure(action)
        if pat == "type_mismatch":
            return self._type_mismatch_failure(action)
        if pat == "condition_flip":
            return self._condition_flip_failure(action)
        if pat == "off_by_one":
            return self._off_by_one_failure(action)
        if action.drift_type == "rename":
            return self._rename_failure(action)
        if action.drift_type == "removal":
            return self._removal_failure(action)
        if action.drift_type == "contract":
            return self._contract_failure(action)
        return None

    def _rename_failure(self, action: DriftAction) -> TestFailure:
        stale = action.stale_ref.split("(")[0]
        info = _RENAME_TESTS.get(stale) or _RENAME_TESTS.get(action.current_ref.split("(")[0])
        if info:
            suite, test, caller = info
        else:
            suite = "test_core"
            test = f"test_{stale.lower()}_call"
            caller = "process_feature"

        return TestFailure(
            test_suite=suite,
            test_name=test,
            error_type="AttributeError",
            error_message=f"module has no attribute '{stale}'",
            call_chain=[test, caller, stale],
        )

    def _removal_failure(self, action: DriftAction) -> TestFailure:
        info = _REMOVAL_TESTS.get(action.stale_ref)
        if info:
            suite, test, caller, mod = info
        else:
            mod = action.metadata.get("module", action.stale_ref.replace("/", ".").replace(".py", ""))
            slug = mod.split(".")[-1]
            suite = "test_imports"
            test = f"test_{slug}_import"
            caller = "init_services"

        return TestFailure(
            test_suite=suite,
            test_name=test,
            error_type="ModuleNotFoundError",
            error_message=f"No module named '{mod}'",
            call_chain=[test, caller, f"import {mod}"],
        )

    def _contract_failure(self, action: DriftAction) -> TestFailure:
        fn = action.metadata.get("function", "unknown")
        old_params = action.metadata.get("old_params") or []
        new_params = action.metadata.get("new_params") or []
        info = _CONTRACT_TESTS.get(fn)
        if info:
            suite, test, caller = info
        else:
            suite = "test_api"
            test = f"test_{fn.lower()}_call"
            caller = "api_handler"

        added = [p for p in new_params if p not in old_params]
        added_str = ", ".join(f"'{p}'" for p in added) if added else "required argument"
        old_call = f"{fn}({', '.join(old_params)})"

        return TestFailure(
            test_suite=suite,
            test_name=test,
            error_type="TypeError",
            error_message=f"{fn}() missing required argument: {added_str}",
            call_chain=[test, caller, old_call],
        )

    # ── New realistic pattern failures ────────────────────────────────────────

    def _partial_rename_failure(self, action: DriftAction) -> TestFailure:
        """Only the stale_context test fails; fresh_context works fine."""
        stale = action.stale_ref.split("(")[0]
        stale_ctx = action.metadata.get("stale_context", "legacy_path")
        test = f"test_{stale_ctx.lower()}_uses_{stale.lower()}"
        caller = stale_ctx
        info = _RENAME_TESTS.get(stale) or _RENAME_TESTS.get(action.current_ref.split("(")[0])
        suite = info[0] if info else "test_core"
        return TestFailure(
            test_suite=suite,
            test_name=test,
            error_type="AttributeError",
            error_message=f"module has no attribute '{stale}' (renamed to {action.current_ref})",
            call_chain=[test, caller, stale],
        )

    def _null_missing_failure(self, action: DriftAction) -> TestFailure:
        """NoneType attribute access because caller doesn't guard for Optional return."""
        fn = action.metadata.get("function", "get_data")
        attr = action.metadata.get("nullable_attribute", "value")
        suite = action.metadata.get("test_suite", "test_core")
        test = action.metadata.get("test_name", f"test_{fn.lower()}_result")
        caller = action.metadata.get("caller", "service_layer")
        return TestFailure(
            test_suite=suite,
            test_name=test,
            error_type="AttributeError",
            error_message=f"'NoneType' object has no attribute '{attr}'",
            call_chain=[test, caller, f"{fn}().{attr}"],
        )

    def _type_mismatch_failure(self, action: DriftAction) -> TestFailure:
        """TypeError because old type (int) passed where new type (str) expected."""
        fn = action.metadata.get("function", "process")
        param = action.metadata.get("param", "id")
        old_type = action.metadata.get("old_type", "int")
        new_type = action.metadata.get("new_type", "str")
        old_example = action.metadata.get("old_example", "123")
        suite = action.metadata.get("test_suite", "test_api")
        test = action.metadata.get("test_name", f"test_{fn.lower()}_call")
        caller = action.metadata.get("caller", "api_handler")
        return TestFailure(
            test_suite=suite,
            test_name=test,
            error_type="TypeError",
            error_message=f"{fn}(): {param} must be {new_type}, got {old_type} ({old_example!r})",
            call_chain=[test, caller, f"{fn}({param}={old_example})"],
        )

    def _condition_flip_failure(self, action: DriftAction) -> TestFailure:
        """AssertionError because boolean flag meaning was inverted."""
        fn = action.metadata.get("function", "validate")
        param = action.metadata.get("param", "strict")
        old_value = action.metadata.get("old_value", "True")
        new_semantics = action.metadata.get("new_semantics", "semantics changed")
        suite = action.metadata.get("test_suite", "test_core")
        test = action.metadata.get("test_name", f"test_{fn.lower()}_behavior")
        caller = action.metadata.get("caller", "handler")
        return TestFailure(
            test_suite=suite,
            test_name=test,
            error_type="AssertionError",
            error_message=f"expected strict=False behavior but got {param}={old_value} ({new_semantics})",
            call_chain=[test, caller, f"{fn}(..., {param}={old_value})"],
        )

    def _off_by_one_failure(self, action: DriftAction) -> TestFailure:
        """IndexError or wrong result because 1-based index passed to 0-based API."""
        fn = action.metadata.get("function", "get_page")
        param = action.metadata.get("param", "page")
        old_call = action.metadata.get("old_call", f"{fn}({param}=1)")
        new_convention = action.metadata.get("new_convention", "0-based")
        suite = action.metadata.get("test_suite", "test_core")
        test = action.metadata.get("test_name", f"test_{fn.lower()}_first_item")
        caller = action.metadata.get("caller", "list_handler")
        return TestFailure(
            test_suite=suite,
            test_name=test,
            error_type="IndexError",
            error_message=f"{fn}(): {param} is {new_convention}; received 1-based value",
            call_chain=[test, caller, old_call],
        )
