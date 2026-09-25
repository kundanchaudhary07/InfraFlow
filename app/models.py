"""
Pydantic models (schemas) for the Order Service.

These models define the shape of data flowing in and out of the API.
They are the single source of truth for request/response validation and
OpenAPI schema generation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class OrderStatus(str, Enum):
    """Lifecycle states a customer order may occupy."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class OrderItem(BaseModel):
    """A single line-item within an order."""

    product_id: str = Field(..., description="SKU / product identifier")
    product_name: str = Field(..., description="Human-readable product name")
    quantity: int = Field(..., ge=1, description="Number of units ordered")
    unit_price: float = Field(..., ge=0.0, description="Price per unit in GBP")

    @property
    def subtotal(self) -> float:
        """Calculated line-item total."""
        return round(self.quantity * self.unit_price, 2)


class Order(BaseModel):
    """Full representation of a customer order."""

    order_id: str = Field(..., description="Unique order identifier")
    customer_id: str = Field(..., description="Unique customer identifier")
    customer_name: str = Field(..., description="Customer display name")
    status: OrderStatus = Field(..., description="Current order lifecycle status")
    items: list[OrderItem] = Field(..., description="Line items in this order")
    total_amount: float = Field(..., ge=0.0, description="Order total in GBP")
    created_at: datetime = Field(..., description="ISO-8601 creation timestamp")
    updated_at: datetime = Field(..., description="ISO-8601 last-updated timestamp")
    notes: Optional[str] = Field(None, description="Optional internal notes")


class OrderListResponse(BaseModel):
    """Envelope for the list-orders endpoint."""

    orders: list[Order]
    count: int = Field(..., description="Total number of orders returned")


class ErrorResponse(BaseModel):
    """Standard error envelope returned on non-2xx responses."""

    detail: str = Field(..., description="Human-readable error message")
