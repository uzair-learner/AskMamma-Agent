"""Seed the local inventory database with realistic demo data."""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from database import get_connection, reset_database, utc_now  # noqa: E402
from rag import reindex_documents  # noqa: E402


SUPPLIERS = [
    ("Northwind Office Supply", "orders@northwind.example", "555-0101", "USA", 5),
    ("FreshPack Foods", "support@freshpack.example", "555-0102", "Canada", 3),
    ("TechCore Components", "sales@techcore.example", "555-0103", "USA", 10),
    ("Global Paper Co", "hello@globalpaper.example", "555-0104", "USA", 7),
    ("Warehouse Essentials", "ops@warehouseessentials.example", "555-0105", "Mexico", 6),
    ("MedSupply Direct", "care@medsupply.example", "555-0106", "USA", 4),
]


PRODUCTS = [
    ("OFF-001", "Copy Paper A4", "Office", "A4 multipurpose paper, 500 sheets", 4, 8.99, 4.1, 12, 30, 120, "A1", None),
    ("OFF-002", "Blue Ballpoint Pens", "Office", "Box of 50 blue pens", 1, 12.5, 5.2, 0, 20, 80, "A2", None),
    ("OFF-003", "Stapler Heavy Duty", "Office", "Metal stapler for office desks", 1, 18.0, 9.0, 45, 15, 50, "A3", None),
    ("OFF-004", "Thermal Receipt Rolls", "Office", "POS thermal paper rolls", 4, 34.0, 18.0, 16, 25, 100, "A4", None),
    ("FOOD-001", "Organic Tomato Sauce", "Food", "Shelf-stable tomato sauce jars", 2, 4.75, 2.1, 9, 20, 90, "B1", "2026-12-01"),
    ("FOOD-002", "Dried Pasta Fusilli", "Food", "Bulk fusilli pasta bags", 2, 6.2, 2.8, 110, 40, 120, "B2", "2027-02-01"),
    ("FOOD-003", "Olive Oil 1L", "Food", "Extra virgin olive oil bottles", 2, 14.5, 8.0, 21, 25, 75, "B3", "2027-01-15"),
    ("FOOD-004", "Mozzarella Packs", "Food", "Refrigerated mozzarella", 2, 5.8, 3.0, 0, 18, 72, "C1", "2026-07-10"),
    ("TECH-001", "USB-C Cable 2m", "Electronics", "Braided USB-C charging cable", 3, 9.99, 3.6, 160, 50, 150, "D1", None),
    ("TECH-002", "Wireless Mouse", "Electronics", "Ergonomic wireless mouse", 3, 24.99, 11.0, 14, 25, 80, "D2", None),
    ("TECH-003", "Laptop Stand", "Electronics", "Adjustable aluminum laptop stand", 3, 39.0, 18.0, 60, 20, 70, "D3", None),
    ("TECH-004", "Barcode Scanner", "Electronics", "USB barcode scanner", 3, 89.0, 44.0, 5, 10, 30, "D4", None),
    ("WH-001", "Packing Tape", "Warehouse", "Clear packing tape rolls", 5, 3.5, 1.1, 22, 40, 150, "E1", None),
    ("WH-002", "Shipping Boxes Small", "Warehouse", "Small corrugated boxes", 5, 1.25, 0.45, 300, 150, 300, "E2", None),
    ("WH-003", "Shipping Boxes Large", "Warehouse", "Large corrugated boxes", 5, 2.75, 1.1, 75, 100, 250, "E3", None),
    ("WH-004", "Bubble Wrap Roll", "Warehouse", "Protective bubble wrap", 5, 22.0, 10.0, 8, 12, 40, "E4", None),
    ("MED-001", "Nitrile Gloves Medium", "Medical", "Box of 100 nitrile gloves", 6, 11.5, 5.8, 19, 25, 100, "F1", "2028-01-01"),
    ("MED-002", "Hand Sanitizer 500ml", "Medical", "Alcohol hand sanitizer", 6, 7.25, 2.9, 0, 35, 140, "F2", "2027-04-01"),
    ("MED-003", "Disposable Masks", "Medical", "Box of 50 masks", 6, 9.5, 4.0, 220, 80, 200, "F3", "2028-06-01"),
    ("MED-004", "First Aid Kit", "Medical", "Workplace first aid kit", 6, 32.0, 16.0, 11, 12, 40, "F4", "2029-01-01"),
    ("FOOD-005", "Basil Pesto", "Food", "Prepared basil pesto jars", 2, 6.8, 3.2, 18, 20, 80, "B4", "2026-10-01"),
    ("OFF-005", "Printer Toner Black", "Office", "Laser printer toner cartridge", 1, 79.0, 41.0, 7, 10, 25, "A5", None),
]


def seed() -> None:
    reset_database()
    now = utc_now()
    with get_connection() as connection:
        connection.executemany(
            """
            INSERT INTO suppliers (name, contact_email, phone, country, lead_time_days, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [(name, email, phone, country, lead, now, now) for name, email, phone, country, lead in SUPPLIERS],
        )
        connection.executemany(
            """
            INSERT INTO products (
                sku, name, category, description, supplier_id, price, cost, stock_quantity,
                reorder_level, reorder_quantity, location, expiry_date, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(*product, now, now) for product in PRODUCTS],
        )
        products = connection.execute("SELECT id, price FROM products").fetchall()
        start = date.today() - timedelta(days=365)
        rows = []
        random.seed(7)
        for product in products:
            for month in range(12):
                sale_day = start + timedelta(days=month * 30 + random.randint(0, 20))
                base = random.randint(8, 55)
                if product["id"] in {1, 5, 10, 13, 18, 22}:
                    base += month * 3
                quantity = max(1, base)
                rows.append((product["id"], quantity, sale_day.isoformat(), round(quantity * product["price"], 2), random.choice(["online", "retail", "wholesale"])))
        connection.executemany(
            """
            INSERT INTO sales_history (product_id, quantity_sold, sale_date, revenue, channel)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
    reindex_documents()
    print("Seeded inventory database and indexed sample documents.")


if __name__ == "__main__":
    seed()
