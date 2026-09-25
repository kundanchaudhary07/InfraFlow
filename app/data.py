"""
In-memory order store.

This module provides a small, realistic set of sample orders that the API
serves during Phase 1.  There is no database; all data lives in this module
and is rebuilt on every process start.

Design rationale
----------------
- Keeping data out of the route files keeps routes thin and focused on
  HTTP concerns only.
- The helper ``get_order_by_id`` encapsulates the lookup so routes do not
  need to understand the storage layout.
- Using a dict for O(1) lookup is more realistic than scanning a list.
"""

from datetime import datetime, timezone
from typing import Optional

from app.models import Order, OrderItem, OrderStatus


def _ts(date_str: str) -> datetime:
    """Parse an ISO-8601 date string to a timezone-aware datetime (UTC)."""
    return datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

_SAMPLE_ORDERS: list[Order] = [
    Order(
        order_id="ORD-0001",
        customer_id="CUST-101",
        customer_name="Alice Pemberton",
        status=OrderStatus.DELIVERED,
        items=[
            OrderItem(
                product_id="SKU-LAPTOP-PRO",
                product_name="LaptopPro 15\" (2025)",
                quantity=1,
                unit_price=1299.99,
            ),
            OrderItem(
                product_id="SKU-MOUSE-WL",
                product_name="Wireless Ergonomic Mouse",
                quantity=1,
                unit_price=49.99,
            ),
        ],
        total_amount=1349.98,
        created_at=_ts("2026-08-01T09:15:00"),
        updated_at=_ts("2026-08-05T14:30:00"),
    ),
    Order(
        order_id="ORD-0002",
        customer_id="CUST-102",
        customer_name="Ben Hartley",
        status=OrderStatus.PROCESSING,
        items=[
            OrderItem(
                product_id="SKU-KEYBOARD-MECH",
                product_name="Mechanical Keyboard TKL",
                quantity=1,
                unit_price=129.99,
            ),
            OrderItem(
                product_id="SKU-MONITOR-27",
                product_name='27" 4K Monitor',
                quantity=2,
                unit_price=449.99,
            ),
        ],
        total_amount=1029.97,
        created_at=_ts("2026-09-10T11:00:00"),
        updated_at=_ts("2026-09-11T08:45:00"),
    ),
    Order(
        order_id="ORD-0003",
        customer_id="CUST-103",
        customer_name="Clara Novak",
        status=OrderStatus.PENDING,
        items=[
            OrderItem(
                product_id="SKU-WEBCAM-HD",
                product_name="1080p HD Webcam",
                quantity=1,
                unit_price=79.99,
            ),
        ],
        total_amount=79.99,
        created_at=_ts("2026-09-13T22:00:00"),
        updated_at=_ts("2026-09-13T22:00:00"),
        notes="Customer requested express delivery.",
    ),
    Order(
        order_id="ORD-0004",
        customer_id="CUST-104",
        customer_name="David Osei",
        status=OrderStatus.SHIPPED,
        items=[
            OrderItem(
                product_id="SKU-HEADSET-PRO",
                product_name="Noise-Cancelling Headset",
                quantity=1,
                unit_price=299.99,
            ),
            OrderItem(
                product_id="SKU-USB-HUB",
                product_name="7-Port USB-C Hub",
                quantity=1,
                unit_price=39.99,
            ),
        ],
        total_amount=339.98,
        created_at=_ts("2026-09-09T16:30:00"),
        updated_at=_ts("2026-09-12T10:00:00"),
    ),
    Order(
        order_id="ORD-0005",
        customer_id="CUST-105",
        customer_name="Eva Rossi",
        status=OrderStatus.CANCELLED,
        items=[
            OrderItem(
                product_id="SKU-TABLET-AIR",
                product_name="TabletAir 11 (256GB)",
                quantity=1,
                unit_price=699.99,
            ),
        ],
        total_amount=699.99,
        created_at=_ts("2026-09-07T13:00:00"),
        updated_at=_ts("2026-09-08T09:15:00"),
        notes="Cancelled by customer — duplicate order.",
    ),
]

# Dict for O(1) lookup by order_id
_ORDER_INDEX: dict[str, Order] = {order.order_id: order for order in _SAMPLE_ORDERS}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_all_orders() -> list[Order]:
    """Return all sample orders."""
    return _SAMPLE_ORDERS


def get_order_by_id(order_id: str) -> Optional[Order]:
    """Return an order by its ID, or None if not found."""
    return _ORDER_INDEX.get(order_id)
