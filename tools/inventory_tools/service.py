"""Inventory-facing tools."""

from __future__ import annotations

from typing import Any

from db.database import (
    find_product,
    get_connection,
    get_product,
    list_products,
    low_stock_products,
    out_of_stock_products,
    rows_to_dicts,
    utc_now,
)


def demo_item_lookup(query: str) -> list[dict[str, Any]]:
    return list_products(search=query, limit=25)


def demo_status(identifier: str | None = None) -> dict[str, Any]:
    if identifier:
        item = find_product(identifier)
        if not item:
            return {"found": False, "message": f"No sample demo item found for `{identifier}`."}
        return {
            "found": True,
            "item": item,
            "low_stock": 0 < item["stock_quantity"] <= item["reorder_level"],
            "out_of_stock": item["stock_quantity"] <= 0,
        }
    return {"found": True, "low_stock": low_stock_products(), "out_of_stock": out_of_stock_products()}


def demo_partner_lookup(identifier: str) -> dict[str, Any]:
    item = find_product(identifier)
    if not item:
        return {"found": False, "message": f"No sample demo item found for `{identifier}`."}
    return {
        "found": True,
        "item": item["name"],
        "partner": {
            "name": item.get("supplier_name"),
            "email": item.get("contact_email"),
            "lead_time_days": item.get("lead_time_days"),
        },
    }


def add_demo_movement(product_id: int, movement_type: str, quantity: int, reason: str = "", confirm: bool = False) -> dict[str, Any]:
    if not confirm:
        raise ValueError("Human confirmation is required before writing a demo movement.")
    if movement_type not in {"stock_in", "stock_out", "adjustment"}:
        raise ValueError("movement_type must be stock_in, stock_out, or adjustment")
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    item = get_product(product_id)
    if not item:
        raise ValueError(f"Demo item {product_id} not found")
    delta = quantity if movement_type == "stock_in" else -quantity
    if movement_type == "adjustment":
        delta = quantity - item["stock_quantity"]
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO inventory_movements (product_id, movement_type, quantity, reason, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (product_id, movement_type, quantity, reason, utc_now()),
        )
        connection.execute(
            "UPDATE products SET stock_quantity = stock_quantity + ?, updated_at = ? WHERE id = ?",
            (delta, utc_now(), product_id),
        )
    return {"item": get_product(product_id), "movement_type": movement_type, "quantity": quantity}


def historical_movements(product_id: int | None = None, limit: int = 25) -> list[dict[str, Any]]:
    params: list[Any] = []
    where = ""
    if product_id is not None:
        where = "WHERE m.product_id = ?"
        params.append(product_id)
    params.append(limit)
    with get_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT m.*, p.name, p.sku
            FROM inventory_movements m
            JOIN products p ON p.id = m.product_id
            {where}
            ORDER BY m.id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return rows_to_dicts(rows)

