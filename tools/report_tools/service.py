"""Report generation tools."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from core import config
from db.database import get_connection, utc_now
from tools.forecast_tools.service import demo_forecast, demo_reorder_recommendations
from tools.inventory_tools.service import low_stock_products, out_of_stock_products


def write_demo_report(title: str = "AskMamma Operations Report", output_format: str = "xlsx") -> dict[str, str]:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    low = low_stock_products()
    out = out_of_stock_products()
    recs = demo_reorder_recommendations()
    forecast = demo_forecast(months=6)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"askmamma-report-{timestamp}"
    output_format = output_format.lower()
    path = config.REPORT_DIR / f"{base_name}.{output_format}"
    summary_rows = [
        {"metric": "title", "value": title},
        {"metric": "generated_at", "value": utc_now()},
        {"metric": "low_stock_items", "value": len(low)},
        {"metric": "out_of_stock_items", "value": len(out)},
        {"metric": "forecast_summary", "value": forecast.get("explanation", forecast.get("message"))},
    ]
    if output_format == "json":
        payload = {"summary": summary_rows, "low_stock": low, "out_of_stock": out, "recommendations": recs, "forecast": forecast}
        path.write_text(pd.Series(payload).to_json(indent=2), encoding="utf-8")
    elif output_format == "md":
        lines = [f"# {title}", "", f"Generated: {utc_now()}", "", "## Summary"]
        lines.extend([f"- {row['metric']}: {row['value']}" for row in summary_rows])
        lines.extend(["", "## Reorder Recommendations"])
        lines.extend([f"- {item['sku']} {item['name']}: recommend {item['recommended_quantity']}" for item in recs] or ["- None"])
        path.write_text("\n".join(lines), encoding="utf-8")
    elif output_format == "txt":
        lines = [title, f"Generated: {utc_now()}", ""]
        lines.extend([f"{row['metric']}: {row['value']}" for row in summary_rows])
        path.write_text("\n".join(lines), encoding="utf-8")
    else:
        path = config.REPORT_DIR / f"{base_name}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
            pd.DataFrame(low or [{"message": "No low-availability demo items"}]).to_excel(writer, sheet_name="Low Stock", index=False)
            pd.DataFrame(out or [{"message": "No out-of-stock demo items"}]).to_excel(writer, sheet_name="Out of Stock", index=False)
            pd.DataFrame(recs or [{"message": "No reorder recommendations"}]).to_excel(writer, sheet_name="Reorder Recommendations", index=False)
    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO long_term_memory (memory_key, memory_value, created_at, updated_at)
            VALUES (?, ?, COALESCE((SELECT created_at FROM long_term_memory WHERE memory_key = ?), ?), ?)
            """,
            ("last_report_path", str(path), "last_report_path", utc_now(), utc_now()),
        )
    return {"path": str(path), "file_name": path.name, "summary": f"Saved report to {path}", "download_name": path.name}
