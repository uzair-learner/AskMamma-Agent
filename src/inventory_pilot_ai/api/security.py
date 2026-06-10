"""JWT and password security helpers for the demo API."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from inventory_pilot_ai import config


class TokenError(Exception):
    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        self.message = message
        super().__init__(message)


def hash_password(password: str, *, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 210_000)
    return f"pbkdf2_sha256${salt}${base64.urlsafe_b64encode(digest).decode('ascii')}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, salt, digest = stored_hash.split("$", 2)
    except ValueError:
        return False
    if scheme != "pbkdf2_sha256":
        return False
    return hmac.compare_digest(hash_password(password, salt=salt), stored_hash)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _sign(message: str) -> str:
    if config.JWT_ALGORITHM != "HS256":
        raise TokenError("unsupported_algorithm", "Only HS256 JWTs are supported by this demo API.")
    signature = hmac.new(config.JWT_SECRET.encode("utf-8"), message.encode("ascii"), hashlib.sha256).digest()
    return _b64url(signature)


def create_access_token(*, user: dict[str, Any], session_id: str) -> str:
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=config.JWT_EXPIRY_MINUTES)
    header = {"alg": config.JWT_ALGORITHM, "typ": "JWT"}
    payload = {
        "sub": str(user["id"]),
        "username": user["username"],
        "role": user["role"],
        "tenant_id": int(user["tenant_id"]),
        "sid": session_id,
        "iss": config.JWT_ISSUER,
        "aud": config.JWT_AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
    }
    signing_input = ".".join(
        [
            _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8")),
            _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
        ]
    )
    return f"{signing_input}.{_sign(signing_input)}"


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        header_part, payload_part, signature = token.split(".", 2)
    except ValueError as exc:
        raise TokenError("invalid_token", "Invalid token format.") from exc

    signing_input = f"{header_part}.{payload_part}"
    if not hmac.compare_digest(_sign(signing_input), signature):
        raise TokenError("invalid_token", "Invalid token signature.")

    try:
        header = json.loads(_b64url_decode(header_part))
        payload = json.loads(_b64url_decode(payload_part))
    except (ValueError, json.JSONDecodeError) as exc:
        raise TokenError("invalid_token", "Invalid token payload.") from exc

    if header.get("alg") != config.JWT_ALGORITHM:
        raise TokenError("invalid_token", "Invalid token algorithm.")
    if payload.get("iss") != config.JWT_ISSUER:
        raise TokenError("invalid_token", "Invalid token issuer.")
    if payload.get("aud") != config.JWT_AUDIENCE:
        raise TokenError("invalid_token", "Invalid token audience.")
    if int(payload.get("exp", 0)) < int(datetime.now(timezone.utc).timestamp()):
        raise TokenError("token_expired", "Token has expired.")
    return payload
