"""Tiny orders module used by the V2 arena.

Each public function is mutation-friendly: simple body, predictable inputs
and outputs, and exercised by tests in tests/test_orders.py.
"""

from __future__ import annotations

from typing import Any


def fetchUserData(user_id):
    """Return a dict for a known user id, or None for unknown ids."""
    users = {
        "u1": {"id": "u1", "email": "alice@example.com", "active": True},
        "u2": {"id": "u2", "email": "bob@example.com", "active": False},
    }
    return users.get(user_id)


def createOrder(item, qty):
    """Build a simple order dict from item + qty (legacy 2-arg signature)."""
    return {"item": item, "qty": int(qty), "total": int(qty)}


def validateInput(data, strict):
    """Validate a payload; strict=True rejects unknown fields."""
    allowed = {"item", "qty", "user_id"}
    if strict:
        for k in data:
            if k not in allowed:
                return False
    return True


def getPageItems(items, page):
    """Return items for a 1-based page index of size 2."""
    if page < 1:
        return []
    start = (page - 1) * 2
    return list(items[start:start + 2])


def sendNotification(user_id, message):
    """Build a notification record from int user_id."""
    return {"user_id": int(user_id), "message": str(message)}
