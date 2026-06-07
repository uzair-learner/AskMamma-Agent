"""Small semantic-style memory helpers stored in long_term_memory."""

from __future__ import annotations

from db.database import get_connection, initialize_database, utc_now


def remember(key: str, value: str) -> None:
    initialize_database()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO long_term_memory (memory_key, memory_value, created_at, updated_at)
            VALUES (?, ?, COALESCE((SELECT created_at FROM long_term_memory WHERE memory_key = ?), ?), ?)
            """,
            (key, value, key, utc_now(), utc_now()),
        )


def recall(key: str) -> str | None:
    initialize_database()
    with get_connection() as connection:
        row = connection.execute(
            "SELECT memory_value FROM long_term_memory WHERE memory_key = ?",
            (key,),
        ).fetchone()
    return row["memory_value"] if row else None

