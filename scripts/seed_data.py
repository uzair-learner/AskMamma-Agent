"""Seed the local Inventory Pilot AI database with realistic demo data."""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_pilot_ai.db.database import get_connection, reset_database, utc_now  # noqa: E402
from inventory_pilot_ai.rag.retriever import reindex_documents  # noqa: E402


SUPPLIERS = [
    ("Northwind Office Supply", "orders@northwind.example", "555-0101", "USA", 5),
    ("FreshPack Foods", "support@freshpack.example", "555-0102", "Canada", 3),
    ("TechCore Components", "sales@techcore.example", "555-0103", "USA", 10),
    ("Global Paper Co", "hello@globalpaper.example", "555-0104", "USA", 7),
    ("Warehouse Essentials", "ops@warehouseessentials.example", "555-0105", "Mexico", 6),
    ("MedSupply Direct", "care@medsupply.example", "555-0106", "USA", 4),
    ("CafeOps Provisions", "orders@cafeops.example", "555-0107", "USA", 4),
    ("GreenTable Goods", "supply@greentable.example", "555-0108", "Canada", 5),
    ("SecureClean Services", "dispatch@secureclean.example", "555-0109", "USA", 2),
    ("EventReady Rentals", "events@eventready.example", "555-0110", "USA", 8),
    ("DataDesk Software", "renewals@datadesk.example", "555-0111", "Ireland", 1),
    ("QuickShip Packaging", "care@quickship.example", "555-0112", "USA", 3),
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
    ("OFF-006", "Printer Toner Cyan", "Office", "Laser printer cyan toner cartridge", 1, 82.0, 43.0, 13, 10, 25, "A5", None),
    ("OFF-007", "Printer Toner Magenta", "Office", "Laser printer magenta toner cartridge", 1, 82.0, 43.0, 9, 10, 25, "A5", None),
    ("OFF-008", "Printer Toner Yellow", "Office", "Laser printer yellow toner cartridge", 1, 82.0, 43.0, 18, 10, 25, "A5", None),
    ("OFF-009", "Sticky Notes 3x3", "Office", "Pack of 24 sticky note pads", 1, 14.25, 6.1, 84, 35, 100, "A6", None),
    ("OFF-010", "Whiteboard Markers", "Office", "Assorted dry erase marker set", 1, 16.75, 7.4, 29, 25, 80, "A7", None),
    ("OFF-011", "File Folders Letter", "Office", "Box of 100 letter file folders", 4, 19.5, 8.25, 44, 30, 90, "A8", None),
    ("FOOD-006", "Roasted Red Peppers", "Food", "Jarred roasted red peppers", 8, 5.9, 2.6, 32, 22, 90, "B5", "2026-11-01"),
    ("FOOD-007", "Artichoke Hearts", "Food", "Marinated artichoke hearts", 8, 7.4, 3.4, 14, 20, 70, "B6", "2026-09-15"),
    ("FOOD-008", "Garlic Aioli", "Food", "Prepared garlic aioli bottles", 7, 4.95, 2.0, 6, 18, 60, "C2", "2026-08-15"),
    ("FOOD-009", "Lemon Sparkling Water", "Beverage", "Case of lemon sparkling water", 7, 18.0, 8.2, 95, 45, 140, "C3", "2027-03-01"),
    ("FOOD-010", "Cold Brew Coffee", "Beverage", "Bottled cold brew coffee", 7, 3.95, 1.5, 40, 35, 120, "C4", "2026-09-01"),
    ("FOOD-011", "Plant-Based Protein Mix", "Food", "Shelf-stable plant protein blend", 8, 12.5, 6.0, 24, 20, 65, "B7", "2027-01-20"),
    ("TECH-005", "Tablet Stand", "Electronics", "Countertop tablet stand", 3, 34.0, 15.5, 23, 18, 55, "D5", None),
    ("TECH-006", "Receipt Printer", "Electronics", "Thermal receipt printer", 3, 159.0, 82.0, 4, 8, 20, "D6", None),
    ("TECH-007", "Network Switch 8 Port", "Electronics", "Compact unmanaged switch", 3, 49.0, 21.0, 17, 12, 40, "D7", None),
    ("TECH-008", "Label Printer", "Electronics", "Shipping label printer", 3, 129.0, 65.0, 7, 10, 25, "D8", None),
    ("TECH-009", "POS Cash Drawer", "Electronics", "Cash drawer with RJ11 connection", 3, 119.0, 58.0, 11, 8, 20, "D9", None),
    ("WH-005", "Packing Peanuts", "Warehouse", "Bag of biodegradable packing peanuts", 12, 18.0, 7.5, 60, 40, 140, "E5", None),
    ("WH-006", "Padded Mailers Small", "Warehouse", "Pack of 100 padded mailers", 12, 28.0, 12.5, 140, 75, 180, "E6", None),
    ("WH-007", "Padded Mailers Large", "Warehouse", "Pack of 100 large padded mailers", 12, 36.0, 17.0, 66, 70, 180, "E7", None),
    ("WH-008", "Shipping Labels 4x6", "Warehouse", "Roll of 4x6 shipping labels", 12, 24.0, 9.5, 35, 45, 130, "E8", None),
    ("WH-009", "Pallet Wrap", "Warehouse", "Industrial pallet wrap roll", 5, 31.0, 13.0, 10, 16, 50, "E9", None),
    ("MED-005", "Alcohol Wipes", "Medical", "Box of 200 alcohol wipes", 6, 8.8, 3.1, 52, 40, 120, "F5", "2028-02-01"),
    ("MED-006", "Digital Thermometer", "Medical", "Contact digital thermometer", 6, 14.25, 6.7, 16, 18, 60, "F6", "2028-08-01"),
    ("MED-007", "Eye Wash Bottles", "Medical", "Sterile eye wash bottle", 6, 11.0, 5.4, 21, 20, 60, "F7", "2027-12-01"),
    ("MED-008", "Disinfectant Spray", "Medical", "Surface disinfectant spray", 9, 6.5, 2.7, 28, 30, 100, "F8", "2027-05-01"),
    ("CLEAN-001", "Microfiber Cloths", "Cleaning", "Pack of 50 microfiber cloths", 9, 22.0, 9.0, 80, 45, 120, "G1", None),
    ("CLEAN-002", "Floor Cleaner Concentrate", "Cleaning", "Commercial floor cleaner", 9, 27.5, 12.25, 19, 20, 70, "G2", "2027-06-01"),
    ("CLEAN-003", "Compostable Trash Bags", "Cleaning", "Box of compostable trash bags", 9, 29.0, 13.0, 36, 35, 110, "G3", None),
    ("CLEAN-004", "Paper Towels", "Cleaning", "Case of paper towel rolls", 4, 41.0, 19.0, 26, 30, 100, "G4", None),
    ("EVENT-001", "Folding Table", "Events", "Six-foot folding table", 10, 74.0, 38.0, 15, 10, 25, "H1", None),
    ("EVENT-002", "Stacking Chair", "Events", "Black stacking chair", 10, 21.0, 9.5, 88, 50, 150, "H2", None),
    ("EVENT-003", "Table Linens White", "Events", "White reusable table linens", 10, 16.0, 7.2, 42, 35, 100, "H3", None),
    ("EVENT-004", "Portable Sign Stand", "Events", "Adjustable sign stand", 10, 32.0, 14.5, 12, 15, 45, "H4", None),
    ("SOFT-001", "Analytics Dashboard License", "Software", "Monthly analytics dashboard license", 11, 59.0, 18.0, 999, 100, 200, "S1", None),
    ("SOFT-002", "Scheduling App License", "Software", "Monthly staff scheduling app license", 11, 39.0, 12.0, 999, 100, 200, "S2", None),
    ("SOFT-003", "Document OCR Credits", "Software", "Pack of 1000 OCR processing credits", 11, 25.0, 8.0, 240, 120, 300, "S3", None),
    ("SOFT-004", "Support Ticket Bundle", "Software", "Bundle of 50 support tickets", 11, 99.0, 40.0, 75, 40, 120, "S4", None),
    ("OPS-001", "Employee Badge Lanyards", "Operations", "Pack of branded lanyards", 5, 18.5, 7.0, 65, 35, 100, "I1", None),
    ("OPS-002", "Visitor Badges", "Operations", "Roll of visitor badge labels", 12, 21.0, 8.5, 31, 30, 90, "I2", None),
    ("OPS-003", "Tamper Evident Seals", "Operations", "Pack of serialized safety seals", 12, 44.0, 20.0, 13, 20, 80, "I3", None),
    ("OPS-004", "Receiving Clipboard", "Operations", "Durable receiving clipboard", 5, 12.0, 4.8, 27, 18, 60, "I4", None),
]


MOVEMENT_REASONS = [
    "initial receiving",
    "cycle count correction",
    "sample order fulfillment",
    "demo restock",
    "quality hold release",
    "damaged item adjustment",
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
        products = connection.execute("SELECT id, sku, category, price, stock_quantity FROM products").fetchall()
        start = date.today() - timedelta(days=540)
        sales_rows = []
        movement_rows = []
        random.seed(7)
        for product in products:
            for month in range(18):
                category_boost = {
                    "Food": 18,
                    "Beverage": 16,
                    "Warehouse": 12,
                    "Cleaning": 10,
                    "Office": 8,
                    "Electronics": 5,
                    "Medical": 7,
                    "Events": 6,
                    "Software": 14,
                    "Operations": 7,
                }.get(product["category"], 6)
                trend = month * 2 if product["sku"] in {"WH-001", "FOOD-001", "TECH-002", "CLEAN-004", "SOFT-003", "OPS-003"} else month // 3
                seasonal = 10 if month in {2, 5, 8, 11, 14, 17} and product["category"] in {"Events", "Beverage", "Cleaning"} else 0
                for _ in range(2):
                    sale_day = start + timedelta(days=month * 30 + random.randint(0, 26))
                    quantity = max(1, random.randint(3, 28) + category_boost + trend + seasonal)
                    sales_rows.append(
                        (
                            product["id"],
                            quantity,
                            sale_day.isoformat(),
                            round(quantity * product["price"], 2),
                            random.choice(["online", "retail", "wholesale", "marketplace", "support"]),
                        )
                    )

            movement_rows.append((product["id"], "stock_in", max(20, product["stock_quantity"] + random.randint(10, 90)), MOVEMENT_REASONS[0], utc_now()))
            if product["stock_quantity"] < 25:
                movement_rows.append((product["id"], "adjustment", product["stock_quantity"], random.choice(MOVEMENT_REASONS[1:]), utc_now()))
            else:
                movement_rows.append((product["id"], "stock_out", random.randint(1, 12), random.choice(MOVEMENT_REASONS[2:]), utc_now()))
        connection.executemany(
            """
            INSERT INTO sales_history (product_id, quantity_sold, sale_date, revenue, channel)
            VALUES (?, ?, ?, ?, ?)
            """,
            sales_rows,
        )
        connection.executemany(
            """
            INSERT INTO inventory_movements (product_id, movement_type, quantity, reason, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            movement_rows,
        )
    reindex_documents()
    print(
        f"Seeded Inventory Pilot AI database with {len(SUPPLIERS)} suppliers, "
        f"{len(PRODUCTS)} products, {len(sales_rows)} sales rows, "
        f"{len(movement_rows)} movement rows, and indexed sample documents."
    )


if __name__ == "__main__":
    seed()
