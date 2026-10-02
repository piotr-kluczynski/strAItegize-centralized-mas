from pydantic import BaseModel, Field
from typing import Literal
from strAItegize_common.others.order import Order

def describe_order(order: Order) -> str:
    if order.order_type == "move":
        return (
            f"move "
            f"(unit_id: {order.unit_id}; "
            f"move_vector: {order.move_q}, {order.move_r}, {order.move_s})"
        )

    elif order.order_type == "attack":
        return (
            f"attack "
            f"(unit_id: {order.unit_id}; "
            f"target_unit: {order.target_id})"
        )

    elif order.order_type == "support":
        return (
            f"support "
            f"(unit_id: {order.unit_id}; "
            f"target_unit: {order.target_id})"
        )

    elif order.order_type == "hold":
        return f"hold (unit_id: {order.unit_id})"

    else:
        raise ValueError(f"Unknown order type: {order.order_type}")