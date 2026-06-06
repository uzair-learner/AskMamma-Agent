"""Small local cart API compatible with the add_to_cart tool.

This is a no-dependency fallback for environments where json-server is not
available or does not stay running in the background.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse


HOST = "localhost"
PORT = 3000
DATA_PATH = Path(__file__).with_name("cart-data.json")


def read_cart() -> dict:
    if not DATA_PATH.exists():
        DATA_PATH.write_text(json.dumps({"cart": []}, indent=2), encoding="utf-8")
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def write_cart(data: dict) -> None:
    DATA_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


class CartRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if urlparse(self.path).path != "/cart":
            self._send_json(404, {"error": "Not found"})
            return
        self._send_json(200, read_cart().get("cart", []))

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/cart":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            item = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(item.get("name"), str):
                raise ValueError("name must be a string")
            item["price"] = float(item["price"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self._send_json(400, {"error": f"Invalid cart item: {exc}"})
            return

        data = read_cart()
        cart = data.setdefault("cart", [])
        item.setdefault("id", len(cart) + 1)
        cart.append(item)
        write_cart(data)
        self._send_json(201, item)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    server = HTTPServer((HOST, PORT), CartRequestHandler)
    print(f"Cart API running at http://{HOST}:{PORT}/cart")
    server.serve_forever()


if __name__ == "__main__":
    main()
