"""Unit tests for src.orders — these are the ground-truth oracle for V2."""

from __future__ import annotations

from src.orders import (
    createOrder,
    fetchUserData,
    getPageItems,
    sendNotification,
    validateInput,
)


def test_fetch_user_data_returns_known_user():
    user = fetchUserData("u1")
    assert user is not None
    assert user["email"] == "alice@example.com"


def test_fetch_user_data_unknown_returns_none():
    assert fetchUserData("missing") is None


def test_create_order_two_args():
    order = createOrder("widget", 3)
    assert order["item"] == "widget"
    assert order["qty"] == 3


def test_validate_input_strict_rejects_unknown():
    assert validateInput({"item": "x", "qty": 1}, strict=True) is True
    assert validateInput({"item": "x", "weird": 1}, strict=True) is False


def test_get_page_items_first_page():
    items = ["a", "b", "c", "d"]
    assert getPageItems(items, 1) == ["a", "b"]


def test_send_notification_int_user_id():
    rec = sendNotification(7890, "hello")
    assert rec["user_id"] == 7890
    assert rec["message"] == "hello"
