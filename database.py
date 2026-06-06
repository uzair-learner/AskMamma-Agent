"""SQLite data layer for the inventory management agent system."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = ROOT_DIR / "inventory.db"


def _db_path_from_env() -> Path:
    database_url = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH.as_posix()}")
    if database_url.startswith("sqlite:///"):
        return Path(database_url.replace("sqlite:///", "", 1))
    return DB_PATH


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    db_path = _db_path_from_env()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def initialize_database() -> None:
    """Create inventory, memory, tracing, and document tables."""

    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS suppliers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                contact_email TEXT,
                phone TEXT,
                country TEXT,
                lead_time_days INTEGER NOT NULL DEFAULT 7,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT,
                supplier_id INTEGER,
                price REAL NOT NULL,
                cost REAL NOT NULL,
                stock_quantity INTEGER NOT NULL,
                reorder_level INTEGER NOT NULL,
                reorder_quantity INTEGER NOT NULL,
                location TEXT,
                expiry_date TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (supplier_id) REFERENCES suppliers(id)
            );

            CREATE TABLE IF NOT EXISTS inventory_movements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER NOT NULL,
                movement_type TEXT NOT NULL CHECK (movement_type IN ('stock_in', 'stock_out', 'adjustment')),
                quantity INTEGER NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (product_id) REFERENCES products(id)
            );

            CREATE TABLE IF NOT EXISTS sales_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER NOT NULL,
                quantity_sold INTEGER NOT NULL,
                sale_date TEXT NOT NULL,
                revenue REAL NOT NULL,
                channel TEXT,
                FOREIGN KEY (product_id) REFERENCES products(id)
            );

            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                category TEXT,
                forecast_period TEXT NOT NULL,
                predicted_quantity REAL NOT NULL,
                method TEXT NOT NULL,
                explanation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (product_id) REFERENCES products(id)
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_key TEXT NOT NULL UNIQUE,
                memory_value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_input TEXT NOT NULL,
                selected_agent TEXT,
                tools_called TEXT NOT NULL DEFAULT '[]',
                tool_inputs TEXT NOT NULL DEFAULT '[]',
                tool_outputs_summary TEXT NOT NULL DEFAULT '[]',
                final_answer TEXT,
                latency_ms INTEGER,
                errors TEXT,
                token_usage TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                path TEXT NOT NULL,
                content_type TEXT,
                uploaded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_id INTEGER NOT NULL,
                page_number INTEGER,
                text TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );
            """
        )


def reset_database() -> None:
    db_path = _db_path_from_env()
    if db_path.exists():
        db_path.unlink()
    initialize_database()


def list_products(search: str | None = None, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    initialize_database()
    where = ""
    params: list[Any] = []
    if search:
        where = """
        WHERE p.name LIKE ? OR p.sku LIKE ? OR p.category LIKE ? OR p.description LIKE ? OR s.name LIKE ?
        """
        term = f"%{search}%"
        params.extend([term, term, term, term, term])
    params.extend([limit, offset])
    with get_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT p.*, s.name AS supplier_name
            FROM products p
            LEFT JOIN suppliers s ON s.id = p.supplier_id
            {where}
            ORDER BY p.name
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
    return rows_to_dicts(rows)


def get_product(product_id: int) -> dict[str, Any] | None:
    initialize_database()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT p.*, s.name AS supplier_name, s.contact_email, s.lead_time_days
            FROM products p
            LEFT JOIN suppliers s ON s.id = p.supplier_id
            WHERE p.id = ?
            """,
            (product_id,),
        ).fetchone()
    return dict(row) if row else None


def find_product(identifier: str) -> dict[str, Any] | None:
    initialize_database()
    term = f"%{identifier}%"
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT p.*, s.name AS supplier_name, s.contact_email, s.lead_time_days
            FROM products p
            LEFT JOIN suppliers s ON s.id = p.supplier_id
            WHERE p.sku = ? OR p.name LIKE ? OR p.description LIKE ?
            ORDER BY CASE WHEN p.sku = ? THEN 0 ELSE 1 END, p.name
            LIMIT 1
            """,
            (identifier, term, term, identifier),
        ).fetchone()
    return dict(row) if row else None


def create_product(payload: dict[str, Any]) -> dict[str, Any]:
    initialize_database()
    now = utc_now()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO products (
                sku, name, category, description, supplier_id, price, cost,
                stock_quantity, reorder_level, reorder_quantity, location,
                expiry_date, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["sku"],
                payload["name"],
                payload["category"],
                payload.get("description", ""),
                payload.get("supplier_id"),
                payload["price"],
                payload["cost"],
                payload["stock_quantity"],
                payload["reorder_level"],
                payload["reorder_quantity"],
                payload.get("location"),
                payload.get("expiry_date"),
                now,
                now,
            ),
        )
        product_id = cursor.lastrowid
    product = get_product(int(product_id))
    if not product:
        raise RuntimeError("Product creation failed")
    return product


def update_product(product_id: int, payload: dict[str, Any]) -> dict[str, Any] | None:
    initialize_database()
    current = get_product(product_id)
    if not current:
        return None
    fields = [
        "sku",
        "name",
        "category",
        "description",
        "supplier_id",
        "price",
        "cost",
        "stock_quantity",
        "reorder_level",
        "reorder_quantity",
        "location",
        "expiry_date",
    ]
    updates = {field: payload[field] for field in fields if field in payload}
    if not updates:
        return current
    updates["updated_at"] = utc_now()
    assignments = ", ".join(f"{field} = ?" for field in updates)
    with get_connection() as connection:
        connection.execute(
            f"UPDATE products SET {assignments} WHERE id = ?",
            [*updates.values(), product_id],
        )
    return get_product(product_id)


def delete_product(product_id: int) -> bool:
    initialize_database()
    with get_connection() as connection:
        cursor = connection.execute("DELETE FROM products WHERE id = ?", (product_id,))
    return cursor.rowcount > 0


def low_stock_products() -> list[dict[str, Any]]:
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT p.*, s.name AS supplier_name
            FROM products p
            LEFT JOIN suppliers s ON s.id = p.supplier_id
            WHERE p.stock_quantity > 0 AND p.stock_quantity <= p.reorder_level
            ORDER BY p.stock_quantity ASC, p.name
            """
        ).fetchall()
    return rows_to_dicts(rows)


def out_of_stock_products() -> list[dict[str, Any]]:
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT p.*, s.name AS supplier_name
            FROM products p
            LEFT JOIN suppliers s ON s.id = p.supplier_id
            WHERE p.stock_quantity <= 0
            ORDER BY p.name
            """
        ).fetchall()
    return rows_to_dicts(rows)


def dashboard_stats() -> dict[str, Any]:
    initialize_database()
    with get_connection() as connection:
        total_products = connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        low_stock = connection.execute(
            "SELECT COUNT(*) FROM products WHERE stock_quantity > 0 AND stock_quantity <= reorder_level"
        ).fetchone()[0]
        out_of_stock = connection.execute(
            "SELECT COUNT(*) FROM products WHERE stock_quantity <= 0"
        ).fetchone()[0]
        high_demand = connection.execute(
            """
            SELECT p.name, SUM(s.quantity_sold) AS sold
            FROM sales_history s
            JOIN products p ON p.id = s.product_id
            WHERE date(s.sale_date) >= date('now', '-90 days')
            GROUP BY p.id
            ORDER BY sold DESC
            LIMIT 5
            """
        ).fetchall()
        recent_actions = connection.execute(
            """
            SELECT selected_agent, tools_called, final_answer, created_at
            FROM agent_traces
            ORDER BY id DESC
            LIMIT 5
            """
        ).fetchall()
    return {
        "total_products": total_products,
        "low_stock_products": low_stock,
        "out_of_stock_products": out_of_stock,
        "predicted_high_demand_products": rows_to_dicts(high_demand),
        "recent_ai_actions": rows_to_dicts(recent_actions),
    }


if __name__ == "__main__":
    initialize_database()
    print(f"Initialized database at {_db_path_from_env()}")
