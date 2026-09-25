"""
Order routes.

Provides:
  GET /api/v1/orders           — list all orders
  GET /api/v1/orders/{order_id} — retrieve a single order by ID
"""

from fastapi import APIRouter, HTTPException, Path

from app.data import get_all_orders, get_order_by_id
from app.models import ErrorResponse, Order, OrderListResponse

router = APIRouter(prefix="/api/v1/orders", tags=["Orders"])


@router.get(
    "",
    response_model=OrderListResponse,
    summary="List all orders",
    description="Returns all orders currently held in the in-memory store.",
)
def list_orders() -> OrderListResponse:
    """Return the full list of sample orders."""
    orders = get_all_orders()
    return OrderListResponse(orders=orders, count=len(orders))


@router.get(
    "/{order_id}",
    response_model=Order,
    responses={404: {"model": ErrorResponse, "description": "Order not found"}},
    summary="Retrieve a single order",
    description="Returns the order matching `order_id`, or HTTP 404 if it does not exist.",
)
def get_order(
    order_id: str = Path(..., description="The unique order identifier, e.g. ORD-0001"),
) -> Order:
    """Retrieve an order by its ID."""
    order = get_order_by_id(order_id)
    if order is None:
        raise HTTPException(
            status_code=404,
            detail=f"Order '{order_id}' not found.",
        )
    return order
