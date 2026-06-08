"""SQLite data layer for the AskMamma agent system."""

from __future__ import annotations

import os
import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

from api.security import hash_password


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "askmamma.db"


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


TENANT_TABLES = [
    "suppliers",
    "products",
    "inventory_movements",
    "sales_history",
    "forecasts",
    "chat_history",
    "agent_traces",
    "ai_generation_events",
    "documents",
    "document_chunks",
]


def _column_exists(connection: sqlite3.Connection, table: str, column: str) -> bool:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)


def _add_column_if_missing(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    if not _column_exists(connection, table, column):
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _tenant_clause(tenant_id: int | None, alias: str = "") -> tuple[str, list[Any]]:
    if tenant_id is None:
        return "", []
    prefix = f"{alias}." if alias else ""
    return f"{prefix}tenant_id = ?", [tenant_id]


def _seed_demo_security(connection: sqlite3.Connection) -> None:
    now = utc_now()
    tenants = [
        (1, "tenant-a", "Demo Tenant A"),
        (2, "tenant-b", "Demo Tenant B"),
    ]
    connection.executemany(
        """
        INSERT OR IGNORE INTO tenants (id, slug, name, created_at)
        VALUES (?, ?, ?, ?)
        """,
        [(tenant_id, slug, name, now) for tenant_id, slug, name in tenants],
    )
    users = [
        ("admin@example.com", "Admin User", "admin", 1, "AdminPass123!"),
        ("manager@example.com", "Manager User", "manager", 1, "ManagerPass123!"),
        ("analyst@example.com", "Analyst User", "analyst", 1, "AnalystPass123!"),
        ("viewer@example.com", "Viewer User", "viewer", 1, "ViewerPass123!"),
        ("tenantb-viewer@example.com", "Tenant B Viewer", "viewer", 2, "TenantBPass123!"),
    ]
    for username, full_name, role, tenant_id, password in users:
        connection.execute(
            """
            INSERT OR IGNORE INTO users (
                tenant_id, username, full_name, role, password_hash, is_active, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (tenant_id, username, full_name, role, hash_password(password), now, now),
        )


def _normalize_ai_generation_event(event: dict[str, Any]) -> dict[str, Any]:
    event["llm_used"] = bool(event.get("llm_used"))
    response = event.get("response")
    if event.get("status") == "success" and not (str(response).strip() if response is not None else ""):
        event["status"] = "failed"
        event["llm_used"] = False
        event["error_message"] = event.get("error_message") or "Ollama returned an empty response."
    return event


def initialize_database() -> None:
    """Create AskMamma demo data, memory, tracing, and document tables."""

    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS suppliers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id INTEGER NOT NULL DEFAULT 1,
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
                tenant_id INTEGER NOT NULL DEFAULT 1,
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
                tenant_id INTEGER NOT NULL DEFAULT 1,
                product_id INTEGER NOT NULL,
                movement_type TEXT NOT NULL CHECK (movement_type IN ('stock_in', 'stock_out', 'adjustment')),
                quantity INTEGER NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (product_id) REFERENCES products(id)
            );

            CREATE TABLE IF NOT EXISTS sales_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id INTEGER NOT NULL DEFAULT 1,
                product_id INTEGER NOT NULL,
                quantity_sold INTEGER NOT NULL,
                sale_date TEXT NOT NULL,
                revenue REAL NOT NULL,
                channel TEXT,
                FOREIGN KEY (product_id) REFERENCES products(id)
            );

            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id INTEGER NOT NULL DEFAULT 1,
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
                tenant_id INTEGER NOT NULL DEFAULT 1,
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
                tenant_id INTEGER NOT NULL DEFAULT 1,
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

            CREATE TABLE IF NOT EXISTS ai_generation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id INTEGER NOT NULL DEFAULT 1,
                feature_name TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                llm_used INTEGER NOT NULL DEFAULT 0,
                prompt TEXT NOT NULL,
                response TEXT,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT
            );

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id INTEGER NOT NULL DEFAULT 1,
                file_name TEXT NOT NULL,
                path TEXT NOT NULL,
                content_type TEXT,
                uploaded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id INTEGER NOT NULL DEFAULT 1,
                document_id INTEGER NOT NULL,
                chunk_id INTEGER NOT NULL,
                page_number INTEGER,
                text TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );

            CREATE TABLE IF NOT EXISTS tenants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id INTEGER NOT NULL,
                username TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('admin', 'manager', 'analyst', 'viewer')),
                password_hash TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (tenant_id) REFERENCES tenants(id)
            );

            CREATE TABLE IF NOT EXISTS user_sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                tenant_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                revoked_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (tenant_id) REFERENCES tenants(id)
            );

            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id INTEGER NOT NULL,
                user_id INTEGER,
                action TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT,
                old_value TEXT,
                new_value TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (tenant_id) REFERENCES tenants(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS reorder_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id INTEGER NOT NULL,
                supplier_id INTEGER NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('Draft', 'Submitted', 'Approved', 'Rejected', 'Completed')),
                requested_by_user_id INTEGER,
                notes TEXT,
                items_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (tenant_id) REFERENCES tenants(id),
                FOREIGN KEY (supplier_id) REFERENCES suppliers(id),
                FOREIGN KEY (requested_by_user_id) REFERENCES users(id)
            );
            """
        )
        for table in TENANT_TABLES:
            _add_column_if_missing(connection, table, "tenant_id", "INTEGER NOT NULL DEFAULT 1")
        _seed_demo_security(connection)


def reset_database() -> None:
    with get_connection() as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        for row in rows:
            connection.execute(f"DROP TABLE IF EXISTS {row['name']}")
        connection.execute("PRAGMA foreign_keys = ON")
    initialize_database()


def list_products(search: str | None = None, limit: int = 100, offset: int = 0, tenant_id: int | None = None) -> list[dict[str, Any]]:
    initialize_database()
    clauses: list[str] = []
    params: list[Any] = []
    tenant_where, tenant_params = _tenant_clause(tenant_id, "p")
    if tenant_where:
        clauses.append(tenant_where)
        params.extend(tenant_params)
    if search:
        clauses.append("(p.name LIKE ? OR p.sku LIKE ? OR p.category LIKE ? OR p.description LIKE ? OR s.name LIKE ?)")
        term = f"%{search}%"
        params.extend([term, term, term, term, term])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
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


def get_product(product_id: int, tenant_id: int | None = None) -> dict[str, Any] | None:
    initialize_database()
    tenant_where, tenant_params = _tenant_clause(tenant_id, "p")
    extra_where = f" AND {tenant_where}" if tenant_where else ""
    with get_connection() as connection:
        row = connection.execute(
            f"""
            SELECT p.*, s.name AS supplier_name, s.contact_email, s.lead_time_days
            FROM products p
            LEFT JOIN suppliers s ON s.id = p.supplier_id
            WHERE p.id = ?{extra_where}
            """,
            [product_id, *tenant_params],
        ).fetchone()
    return dict(row) if row else None


def find_product(identifier: str, tenant_id: int | None = None) -> dict[str, Any] | None:
    initialize_database()
    term = f"%{identifier}%"
    tenant_where, tenant_params = _tenant_clause(tenant_id, "p")
    extra_where = f" AND {tenant_where}" if tenant_where else ""
    with get_connection() as connection:
        row = connection.execute(
            f"""
            SELECT p.*, s.name AS supplier_name, s.contact_email, s.lead_time_days
            FROM products p
            LEFT JOIN suppliers s ON s.id = p.supplier_id
            WHERE (p.sku = ? OR p.name LIKE ? OR p.description LIKE ?){extra_where}
            ORDER BY CASE WHEN p.sku = ? THEN 0 ELSE 1 END, p.name
            LIMIT 1
            """,
            [identifier, term, term, *tenant_params, identifier],
        ).fetchone()
    return dict(row) if row else None


def create_product(payload: dict[str, Any], tenant_id: int | None = None) -> dict[str, Any]:
    initialize_database()
    now = utc_now()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO products (
                tenant_id, sku, name, category, description, supplier_id, price, cost,
                stock_quantity, reorder_level, reorder_quantity, location,
                expiry_date, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tenant_id or payload.get("tenant_id") or 1,
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
    product = get_product(int(product_id), tenant_id)
    if not product:
        raise RuntimeError("Product creation failed")
    return product


def update_product(product_id: int, payload: dict[str, Any], tenant_id: int | None = None) -> dict[str, Any] | None:
    initialize_database()
    current = get_product(product_id, tenant_id)
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
            f"UPDATE products SET {assignments} WHERE id = ?" + (" AND tenant_id = ?" if tenant_id is not None else ""),
            [*updates.values(), product_id, *([tenant_id] if tenant_id is not None else [])],
        )
    return get_product(product_id, tenant_id)


def delete_product(product_id: int, tenant_id: int | None = None) -> bool:
    initialize_database()
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM products WHERE id = ?" + (" AND tenant_id = ?" if tenant_id is not None else ""),
            [product_id, *([tenant_id] if tenant_id is not None else [])],
        )
    return cursor.rowcount > 0


def low_stock_products(tenant_id: int | None = None) -> list[dict[str, Any]]:
    initialize_database()
    with get_connection() as connection:
        tenant_where, tenant_params = _tenant_clause(tenant_id, "p")
        extra_where = f" AND {tenant_where}" if tenant_where else ""
        rows = connection.execute(
            f"""
            SELECT p.*, s.name AS supplier_name
            FROM products p
            LEFT JOIN suppliers s ON s.id = p.supplier_id
            WHERE p.stock_quantity > 0 AND p.stock_quantity <= p.reorder_level{extra_where}
            ORDER BY p.stock_quantity ASC, p.name
            """,
            tenant_params,
        ).fetchall()
    return rows_to_dicts(rows)


def out_of_stock_products(tenant_id: int | None = None) -> list[dict[str, Any]]:
    initialize_database()
    with get_connection() as connection:
        tenant_where, tenant_params = _tenant_clause(tenant_id, "p")
        extra_where = f" AND {tenant_where}" if tenant_where else ""
        rows = connection.execute(
            f"""
            SELECT p.*, s.name AS supplier_name
            FROM products p
            LEFT JOIN suppliers s ON s.id = p.supplier_id
            WHERE p.stock_quantity <= 0{extra_where}
            ORDER BY p.name
            """,
            tenant_params,
        ).fetchall()
    return rows_to_dicts(rows)


def list_suppliers(tenant_id: int | None = None) -> list[dict[str, Any]]:
    initialize_database()
    with get_connection() as connection:
        tenant_where, tenant_params = _tenant_clause(tenant_id, "s")
        where = f"WHERE {tenant_where}" if tenant_where else ""
        rows = connection.execute(
            f"""
            SELECT
                s.*,
                COUNT(p.id) AS product_count,
                COALESCE(SUM(CASE WHEN p.stock_quantity <= 0 THEN 1 ELSE 0 END), 0) AS out_of_stock_count,
                COALESCE(SUM(CASE WHEN p.stock_quantity > 0 AND p.stock_quantity <= p.reorder_level THEN 1 ELSE 0 END), 0) AS low_stock_count
            FROM suppliers s
            LEFT JOIN products p ON p.supplier_id = s.id AND p.tenant_id = s.tenant_id
            {where}
            GROUP BY s.id
            ORDER BY s.name
            """,
            tenant_params,
        ).fetchall()
    return rows_to_dicts(rows)


def dashboard_stats(tenant_id: int | None = None) -> dict[str, Any]:
    initialize_database()
    tenant_filter = " WHERE tenant_id = ?" if tenant_id is not None else ""
    tenant_params = [tenant_id] if tenant_id is not None else []
    product_prefix = " AND p.tenant_id = ?" if tenant_id is not None else ""
    with get_connection() as connection:
        total_products = connection.execute(f"SELECT COUNT(*) FROM products{tenant_filter}", tenant_params).fetchone()[0]
        low_stock = connection.execute(
            "SELECT COUNT(*) FROM products WHERE stock_quantity > 0 AND stock_quantity <= reorder_level"
            + (" AND tenant_id = ?" if tenant_id is not None else ""),
            tenant_params,
        ).fetchone()[0]
        out_of_stock = connection.execute(
            "SELECT COUNT(*) FROM products WHERE stock_quantity <= 0"
            + (" AND tenant_id = ?" if tenant_id is not None else ""),
            tenant_params,
        ).fetchone()[0]
        high_demand = connection.execute(
            f"""
            SELECT p.name, SUM(s.quantity_sold) AS sold
            FROM sales_history s
            JOIN products p ON p.id = s.product_id
            WHERE date(s.sale_date) >= date('now', '-90 days'){product_prefix}
            GROUP BY p.id
            ORDER BY sold DESC
            LIMIT 5
            """
            ,
            tenant_params,
        ).fetchall()
        recent_actions = connection.execute(
            f"""
            SELECT selected_agent, tools_called, final_answer, created_at
            FROM agent_traces
            {tenant_filter}
            ORDER BY id DESC
            LIMIT 5
            """,
            tenant_params,
        ).fetchall()
    return {
        "total_products": total_products,
        "low_stock_products": low_stock,
        "out_of_stock_products": out_of_stock,
        "predicted_high_demand_products": rows_to_dicts(high_demand),
        "recent_ai_actions": rows_to_dicts(recent_actions),
    }


def log_ai_generation_event(
    *,
    feature_name: str,
    provider: str,
    model: str,
    llm_used: bool,
    prompt: str,
    response: str | None,
    created_at: str,
    status: str,
    error_message: str | None = None,
    tenant_id: int | None = None,
) -> dict[str, Any]:
    initialize_database()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO ai_generation_events (
                tenant_id, feature_name, provider, model, llm_used, prompt, response, created_at, status, error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tenant_id or 1,
                feature_name,
                provider,
                model,
                int(llm_used),
                prompt,
                response,
                created_at,
                status,
                error_message,
            ),
        )
        event_id = int(cursor.lastrowid)
        row = connection.execute(
            "SELECT * FROM ai_generation_events WHERE id = ?",
            (event_id,),
        ).fetchone()
    event = dict(row) if row else {}
    if event:
        event = _normalize_ai_generation_event(event)
    return event


def list_ai_generation_events(limit: int = 50, feature_name: str | None = None, tenant_id: int | None = None) -> list[dict[str, Any]]:
    initialize_database()
    query = """
        SELECT *
        FROM ai_generation_events
    """
    clauses: list[str] = []
    params: list[Any] = []
    if feature_name:
        clauses.append("feature_name = ?")
        params.append(feature_name)
    if tenant_id is not None:
        clauses.append("tenant_id = ?")
        params.append(tenant_id)
    if clauses:
        query += f" WHERE {' AND '.join(clauses)}"
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    events = rows_to_dicts(rows)
    return [_normalize_ai_generation_event(event) for event in events]


def get_user_by_username(username: str) -> dict[str, Any] | None:
    initialize_database()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT u.*, t.slug AS tenant_slug, t.name AS tenant_name
            FROM users u
            JOIN tenants t ON t.id = u.tenant_id
            WHERE lower(u.username) = lower(?)
            """,
            (username,),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    initialize_database()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT u.*, t.slug AS tenant_slug, t.name AS tenant_name
            FROM users u
            JOIN tenants t ON t.id = u.tenant_id
            WHERE u.id = ?
            """,
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def create_user_session(user_id: int, tenant_id: int, expires_minutes: int) -> dict[str, Any]:
    initialize_database()
    session_id = secrets.token_urlsafe(24)
    created_at = utc_now()
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)).isoformat()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO user_sessions (id, user_id, tenant_id, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, user_id, tenant_id, created_at, expires_at),
        )
    return {"id": session_id, "user_id": user_id, "tenant_id": tenant_id, "created_at": created_at, "expires_at": expires_at}


def get_user_session(session_id: str) -> dict[str, Any] | None:
    initialize_database()
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM user_sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def revoke_user_session(session_id: str) -> bool:
    initialize_database()
    with get_connection() as connection:
        cursor = connection.execute(
            "UPDATE user_sessions SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
            (utc_now(), session_id),
        )
    return cursor.rowcount > 0


def log_audit_event(
    *,
    tenant_id: int,
    user_id: int | None,
    action: str,
    entity_type: str,
    entity_id: str | int | None = None,
    old_value: Any = None,
    new_value: Any = None,
) -> dict[str, Any]:
    initialize_database()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO audit_logs (
                tenant_id, user_id, action, entity_type, entity_id, old_value, new_value, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tenant_id,
                user_id,
                action,
                entity_type,
                str(entity_id) if entity_id is not None else None,
                json.dumps(old_value, default=str) if old_value is not None else None,
                json.dumps(new_value, default=str) if new_value is not None else None,
                utc_now(),
            ),
        )
        row = connection.execute("SELECT * FROM audit_logs WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return dict(row) if row else {}


def list_audit_logs(tenant_id: int, limit: int = 100) -> list[dict[str, Any]]:
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT a.*, u.username
            FROM audit_logs a
            LEFT JOIN users u ON u.id = a.user_id
            WHERE a.tenant_id = ?
            ORDER BY a.id DESC
            LIMIT ?
            """,
            (tenant_id, limit),
        ).fetchall()
    return rows_to_dicts(rows)


def create_reorder_request(
    *,
    tenant_id: int,
    supplier_id: int,
    requested_by_user_id: int,
    items: list[dict[str, Any]],
    notes: str = "",
    status: str = "Draft",
) -> dict[str, Any]:
    initialize_database()
    now = utc_now()
    with get_connection() as connection:
        supplier = connection.execute(
            "SELECT id FROM suppliers WHERE id = ? AND tenant_id = ?",
            (supplier_id, tenant_id),
        ).fetchone()
        if not supplier:
            raise ValueError("Supplier not found for current tenant.")
        cursor = connection.execute(
            """
            INSERT INTO reorder_requests (
                tenant_id, supplier_id, status, requested_by_user_id, notes, items_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (tenant_id, supplier_id, status, requested_by_user_id, notes, json.dumps(items, default=str), now, now),
        )
    request = get_reorder_request(int(cursor.lastrowid), tenant_id)
    if not request:
        raise RuntimeError("Reorder request creation failed.")
    return request


def get_reorder_request(request_id: int, tenant_id: int) -> dict[str, Any] | None:
    initialize_database()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT r.*, s.name AS supplier_name, u.username AS requested_by
            FROM reorder_requests r
            JOIN suppliers s ON s.id = r.supplier_id
            LEFT JOIN users u ON u.id = r.requested_by_user_id
            WHERE r.id = ? AND r.tenant_id = ?
            """,
            (request_id, tenant_id),
        ).fetchone()
    if not row:
        return None
    payload = dict(row)
    payload["items"] = json.loads(payload.pop("items_json") or "[]")
    return payload


def list_reorder_requests(tenant_id: int, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    initialize_database()
    params: list[Any] = [tenant_id]
    status_clause = ""
    if status:
        status_clause = " AND r.status = ?"
        params.append(status)
    params.append(limit)
    with get_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT r.*, s.name AS supplier_name, u.username AS requested_by
            FROM reorder_requests r
            JOIN suppliers s ON s.id = r.supplier_id
            LEFT JOIN users u ON u.id = r.requested_by_user_id
            WHERE r.tenant_id = ?{status_clause}
            ORDER BY r.id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    requests = rows_to_dicts(rows)
    for request in requests:
        request["items"] = json.loads(request.pop("items_json") or "[]")
    return requests


def update_reorder_request(request_id: int, tenant_id: int, payload: dict[str, Any]) -> dict[str, Any] | None:
    initialize_database()
    current = get_reorder_request(request_id, tenant_id)
    if not current:
        return None
    allowed = {"status", "notes"}
    updates = {key: value for key, value in payload.items() if key in allowed and value is not None}
    if "items" in payload and payload["items"] is not None:
        updates["items_json"] = json.dumps(payload["items"], default=str)
    if not updates:
        return current
    updates["updated_at"] = utc_now()
    assignments = ", ".join(f"{field} = ?" for field in updates)
    with get_connection() as connection:
        connection.execute(
            f"UPDATE reorder_requests SET {assignments} WHERE id = ? AND tenant_id = ?",
            [*updates.values(), request_id, tenant_id],
        )
    return get_reorder_request(request_id, tenant_id)


if __name__ == "__main__":
    initialize_database()
    print(f"Initialized database at {_db_path_from_env()}")
