"""
Bug Pattern Catalog — the semantic layer above raw drift mutations.

Patterns are derived from common real-world PR bugs:
  partial_rename  — function renamed everywhere except one call site in PR
  null_missing    — return type now Optional; PR doesn't guard against None
  type_mismatch   — parameter type changed (int → str); PR passes old type
  condition_flip  — boolean param semantics inverted; PR passes wrong value
  off_by_one      — index convention changed; PR uses old offset

Legacy patterns (kept for backward compat):
  rename    — function renamed completely
  removal   — file deleted
  contract  — API signature has new required parameter

Usage:
  from env.bug_patterns import BUG_PATTERNS, ALL_PATTERNS
  generator.apply(random.choice(BUG_PATTERNS))
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Pattern definitions ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BugPattern:
    name: str
    description: str
    difficulty: int          # 1 = easy, 2 = medium, 3 = hard (to spot)
    drift_type_group: str    # underlying mutation class for backward compat


# Patterns inspired by real PR bug types
BUG_PATTERNS: list[BugPattern] = [
    BugPattern(
        name="partial_rename",
        description="Function renamed everywhere except one call site; reviewer must spot the lone stale call",
        difficulty=3,
        drift_type_group="rename",
    ),
    BugPattern(
        name="null_missing",
        description="Return type changed to Optional but caller doesn't guard against None",
        difficulty=2,
        drift_type_group="contract",
    ),
    BugPattern(
        name="type_mismatch",
        description="Parameter type changed (int → str); PR still passes the old type",
        difficulty=2,
        drift_type_group="contract",
    ),
    BugPattern(
        name="condition_flip",
        description="Boolean parameter semantics inverted; PR passes the wrong value",
        difficulty=3,
        drift_type_group="contract",
    ),
    BugPattern(
        name="off_by_one",
        description="Index convention changed (1-based → 0-based); PR uses old offset",
        difficulty=2,
        drift_type_group="contract",
    ),
    # Legacy patterns
    BugPattern(
        name="rename",
        description="Function fully renamed; PR uses the old name throughout",
        difficulty=1,
        drift_type_group="rename",
    ),
    BugPattern(
        name="removal",
        description="File/module deleted; PR still imports it",
        difficulty=1,
        drift_type_group="removal",
    ),
    BugPattern(
        name="contract",
        description="API signature has a new required parameter; PR uses old call",
        difficulty=1,
        drift_type_group="contract",
    ),
]

PATTERN_MAP: dict[str, BugPattern] = {p.name: p for p in BUG_PATTERNS}
ALL_PATTERNS: list[str] = [p.name for p in BUG_PATTERNS]
NEW_PATTERNS: list[str] = ["partial_rename", "null_missing", "type_mismatch", "condition_flip", "off_by_one"]
LEGACY_PATTERNS: list[str] = ["rename", "removal", "contract"]

# ── Data catalogs ─────────────────────────────────────────────────────────────

# partial_rename: rename happened, but ONE call site in the PR still uses old name
PARTIAL_RENAME_CASES: list[dict] = [
    {
        "old_name": "getUserData",
        "new_name": "fetchUserData",
        "stale_context": "refresh_cache",
        "fresh_context": "update_profile",
    },
    {
        "old_name": "validateInput",
        "new_name": "sanitizeInput",
        "stale_context": "legacy_endpoint",
        "fresh_context": "new_handler",
    },
    {
        "old_name": "parseResponse",
        "new_name": "deserializeResponse",
        "stale_context": "retry_handler",
        "fresh_context": "primary_handler",
    },
    {
        "old_name": "sendNotification",
        "new_name": "dispatchAlert",
        "stale_context": "fallback_notifier",
        "fresh_context": "main_notifier",
    },
    {
        "old_name": "loadConfig",
        "new_name": "readConfig",
        "stale_context": "dev_bootstrap",
        "fresh_context": "prod_init",
    },
]

# null_missing: function now returns Optional; PR accesses attribute directly
NULL_MISSING_CASES: list[dict] = [
    {
        "function": "getUserData",
        "nullable_attribute": "email",
        "old_return": "UserProfile",
        "new_return": "Optional[UserProfile]",
        "reason": "Deleted accounts now return None instead of raising",
        "test_suite": "test_user_management",
        "test_name": "test_send_welcome_email_to_new_user",
        "caller": "email_dispatcher",
    },
    {
        "function": "loadConfig",
        "nullable_attribute": "database_url",
        "old_return": "Config",
        "new_return": "Optional[Config]",
        "reason": "Missing config file returns None instead of raising",
        "test_suite": "test_config",
        "test_name": "test_db_connect_on_start",
        "caller": "app_init",
    },
    {
        "function": "authenticate",
        "nullable_attribute": "session_token",
        "old_return": "Session",
        "new_return": "Optional[Session]",
        "reason": "Failed auth now returns None instead of raising AuthError",
        "test_suite": "test_auth",
        "test_name": "test_login_then_access_protected",
        "caller": "session_manager",
    },
    {
        "function": "parseResponse",
        "nullable_attribute": "data",
        "old_return": "ParsedResponse",
        "new_return": "Optional[ParsedResponse]",
        "reason": "Malformed responses now return None instead of raising ParseError",
        "test_suite": "test_api",
        "test_name": "test_process_api_response",
        "caller": "api_client",
    },
]

# type_mismatch: parameter type changed; PR still passes old type
TYPE_MISMATCH_CASES: list[dict] = [
    {
        "function": "createOrder",
        "param": "userId",
        "old_type": "int",
        "new_type": "str",
        "old_example": "12345",
        "new_example": "usr_12345",
        "reason": "UUID migration — user IDs are now strings, not integers",
        "test_suite": "test_orders",
        "test_name": "test_create_order_for_existing_user",
        "caller": "checkout_flow",
    },
    {
        "function": "deleteRecord",
        "param": "recordId",
        "old_type": "int",
        "new_type": "str",
        "old_example": "42",
        "new_example": "rec_42",
        "reason": "Record IDs are now prefixed strings for external traceability",
        "test_suite": "test_records",
        "test_name": "test_delete_record_by_id",
        "caller": "record_manager",
    },
    {
        "function": "sendNotification",
        "param": "userId",
        "old_type": "int",
        "new_type": "str",
        "old_example": "7890",
        "new_example": "user_7890",
        "reason": "UUID-based user identifiers across all services",
        "test_suite": "test_notifications",
        "test_name": "test_notify_user_on_event",
        "caller": "notification_service",
    },
    {
        "function": "checkPermission",
        "param": "userId",
        "old_type": "int",
        "new_type": "str",
        "old_example": "101",
        "new_example": "usr_101",
        "reason": "Auth service migrated to string-based user identifiers",
        "test_suite": "test_auth",
        "test_name": "test_permission_check_for_resource",
        "caller": "auth_middleware",
    },
]

# condition_flip: boolean parameter semantics inverted
CONDITION_FLIP_CASES: list[dict] = [
    {
        "function": "validateInput",
        "param": "strict",
        "old_value": "True",
        "new_correct_value": "False",
        "old_semantics": "True = strict validation (reject unknowns)",
        "new_semantics": "True = permissive mode (allow unknowns)",
        "reason": "Flag semantics inverted for backwards-compat mode rollout",
        "test_suite": "test_validation",
        "test_name": "test_form_rejects_unknown_fields",
        "caller": "form_handler",
    },
    {
        "function": "loadConfig",
        "param": "cache",
        "old_value": "True",
        "new_correct_value": "False",
        "old_semantics": "True = use cached config",
        "new_semantics": "True = bypass cache (force refresh)",
        "reason": "Cache parameter inverted to opt-in for freshness by default",
        "test_suite": "test_config",
        "test_name": "test_config_reflects_env_changes",
        "caller": "config_loader",
    },
    {
        "function": "authenticate",
        "param": "remember_me",
        "old_value": "True",
        "new_correct_value": "False",
        "old_semantics": "True = persist session",
        "new_semantics": "True = ephemeral session (auto-logout on close)",
        "reason": "Security policy: sessions are now ephemeral by default",
        "test_suite": "test_auth",
        "test_name": "test_session_persists_across_requests",
        "caller": "auth_controller",
    },
]

# off_by_one: index convention changed
OFF_BY_ONE_CASES: list[dict] = [
    {
        "function": "getLogEntry",
        "param": "line_number",
        "old_convention": "1-based (like editors)",
        "new_convention": "0-based (array index)",
        "old_call": "getLogEntry(line_number=1)",
        "new_correct_call": "getLogEntry(line_number=0)",
        "reason": "Perf optimization: direct array access, no offset conversion",
        "test_suite": "test_logging",
        "test_name": "test_read_first_log_entry",
        "caller": "log_reader",
    },
    {
        "function": "getPageItems",
        "param": "page",
        "old_convention": "1-based page number",
        "new_convention": "0-based page offset",
        "old_call": "getPageItems(page=1)",
        "new_correct_call": "getPageItems(page=0)",
        "reason": "API standardization: offset-based pagination for cursor compat",
        "test_suite": "test_pagination",
        "test_name": "test_list_first_page_of_results",
        "caller": "list_handler",
    },
]
