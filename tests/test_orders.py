"""
Tests for the orders endpoint.

Covers:
  - Listing all orders returns HTTP 200
  - List response contains the expected envelope fields
  - Retrieving a known order returns the correct data
  - Retrieving an unknown order returns HTTP 404 with an error body
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.data import get_all_orders

client = TestClient(app)

KNOWN_ORDER_ID = "ORD-0001"
UNKNOWN_ORDER_ID = "ORD-9999"


# ---------------------------------------------------------------------------
# List orders
# ---------------------------------------------------------------------------


def test_list_orders_returns_200() -> None:
    """GET /api/v1/orders must respond with HTTP 200 OK."""
    response = client.get("/api/v1/orders")
    assert response.status_code == 200


def test_list_orders_response_structure() -> None:
    """Response must contain 'orders' list and matching 'count'."""
    response = client.get("/api/v1/orders")
    body = response.json()

    assert "orders" in body
    assert "count" in body
    assert isinstance(body["orders"], list)
    assert body["count"] == len(body["orders"])


def test_list_orders_count_matches_data() -> None:
    """The count field must match the number of orders in the store."""
    expected = len(get_all_orders())
    response = client.get("/api/v1/orders")
    assert response.json()["count"] == expected


def test_list_orders_items_have_required_fields() -> None:
    """Each order in the list must contain all required Order fields."""
    required_fields = {
        "order_id",
        "customer_id",
        "customer_name",
        "status",
        "items",
        "total_amount",
        "created_at",
        "updated_at",
    }
    response = client.get("/api/v1/orders")
    for order in response.json()["orders"]:
        assert required_fields.issubset(order.keys()), (
            f"Order {order.get('order_id')} is missing required fields"
        )


# ---------------------------------------------------------------------------
# Get single order — found
# ---------------------------------------------------------------------------


def test_get_existing_order_returns_200() -> None:
    """GET /api/v1/orders/{id} for a known order must return HTTP 200."""
    response = client.get(f"/api/v1/orders/{KNOWN_ORDER_ID}")
    assert response.status_code == 200


def test_get_existing_order_returns_correct_id() -> None:
    """The returned order must have the correct order_id."""
    response = client.get(f"/api/v1/orders/{KNOWN_ORDER_ID}")
    assert response.json()["order_id"] == KNOWN_ORDER_ID


def test_get_existing_order_contains_items() -> None:
    """The returned order must include at least one line item."""
    response = client.get(f"/api/v1/orders/{KNOWN_ORDER_ID}")
    assert len(response.json()["items"]) >= 1


# ---------------------------------------------------------------------------
# Get single order — not found
# ---------------------------------------------------------------------------


def test_get_nonexistent_order_returns_404() -> None:
    """GET /api/v1/orders/{id} for an unknown order must return HTTP 404."""
    response = client.get(f"/api/v1/orders/{UNKNOWN_ORDER_ID}")
    assert response.status_code == 404


def test_get_nonexistent_order_returns_error_detail() -> None:
    """404 response must include a 'detail' field with a descriptive message."""
    response = client.get(f"/api/v1/orders/{UNKNOWN_ORDER_ID}")
    body = response.json()
    assert "detail" in body
    assert UNKNOWN_ORDER_ID in body["detail"]


# ---------------------------------------------------------------------------
# Parametrised: all sample orders must be individually retrievable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "order_id",
    [order.order_id for order in get_all_orders()],
)
def test_every_sample_order_is_retrievable(order_id: str) -> None:
    """Every order present in the store must be retrievable by ID."""
    response = client.get(f"/api/v1/orders/{order_id}")
    assert response.status_code == 200
    assert response.json()["order_id"] == order_id
