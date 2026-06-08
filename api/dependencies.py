"""Authentication, tenant, and RBAC dependencies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, Header, HTTPException, status

from api.security import TokenError, decode_access_token
from db.database import get_user_by_id


ROLE_PERMISSIONS = {
    "admin": {
        "admin:read",
        "inventory:read",
        "inventory:write",
        "reorder:read",
        "supplier:read",
        "supplier:write",
        "forecast:read",
        "report:read",
        "report:write",
        "ai:chat",
        "document:read",
        "document:write",
    },
    "manager": {
        "inventory:read",
        "inventory:write",
        "reorder:read",
        "supplier:read",
        "supplier:write",
        "forecast:read",
        "report:read",
        "report:write",
        "ai:chat",
        "document:read",
        "document:write",
    },
    "analyst": {
        "inventory:read",
        "reorder:read",
        "supplier:read",
        "forecast:read",
        "report:read",
        "ai:chat",
        "document:read",
    },
    "viewer": {
        "inventory:read",
        "supplier:read",
    },
}


def _auth_error(error_code: str, message: str, status_code: int = status.HTTP_401_UNAUTHORIZED) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"error_code": error_code, "message": message},
        headers={"WWW-Authenticate": "Bearer"} if status_code == status.HTTP_401_UNAUTHORIZED else None,
    )


def get_current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    if not authorization:
        raise _auth_error("missing_token", "Missing bearer token.")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise _auth_error("missing_token", "Authorization header must be Bearer <token>.")
    try:
        payload = decode_access_token(token)
    except TokenError as exc:
        raise _auth_error(exc.error_code, exc.message) from exc
    user = get_user_by_id(int(payload["sub"]))
    if not user or not user.get("is_active"):
        raise _auth_error("invalid_token", "Authenticated user no longer exists or is inactive.")
    return user


def optional_current_user(authorization: str | None = Header(default=None)) -> dict[str, Any] | None:
    if not authorization:
        return None
    return get_current_user(authorization)


def get_current_tenant_id(user: dict[str, Any] = Depends(get_current_user)) -> int:
    return int(user["tenant_id"])


def require_permission(permission: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def dependency(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        allowed = ROLE_PERMISSIONS.get(user["role"], set())
        if permission not in allowed:
            raise _auth_error(
                "forbidden",
                f"Role '{user['role']}' is not allowed to perform '{permission}'.",
                status.HTTP_403_FORBIDDEN,
            )
        return user

    return dependency
