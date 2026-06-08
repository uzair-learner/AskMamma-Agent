"""Authentication routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies import get_current_user
from api.security import create_access_token, verify_password
from core import config
from db.database import create_user_session, get_user_by_username, log_audit_event, revoke_user_session


router = APIRouter(prefix="/auth", tags=["auth"])


class LoginPayload(BaseModel):
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)


def public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": user["id"],
        "username": user["username"],
        "full_name": user["full_name"],
        "role": user["role"],
        "tenant_id": user["tenant_id"],
        "tenant_slug": user.get("tenant_slug"),
        "tenant_name": user.get("tenant_name"),
    }


@router.post("/login")
def login(payload: LoginPayload) -> dict[str, Any]:
    user = get_user_by_username(payload.username)
    if not user or not user.get("is_active") or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "invalid_credentials", "message": "Invalid username or password."},
        )
    session = create_user_session(user["id"], user["tenant_id"], config.JWT_EXPIRY_MINUTES)
    log_audit_event(
        tenant_id=user["tenant_id"],
        user_id=user["id"],
        action="user_login",
        entity_type="user_session",
        entity_id=session["id"],
        new_value={"username": user["username"]},
    )
    return {
        "access_token": create_access_token(user=user, session_id=session["id"]),
        "token_type": "bearer",
        "expires_in": config.JWT_EXPIRY_MINUTES * 60,
        "user": public_user(user),
    }


@router.get("/me")
def me(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return public_user(user)


@router.post("/logout")
def logout(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    session_id = user.get("session_id")
    if session_id:
        revoke_user_session(str(session_id))
        log_audit_event(
            tenant_id=user["tenant_id"],
            user_id=user["id"],
            action="user_logout",
            entity_type="user_session",
            entity_id=session_id,
        )
    return {"logged_out": True}
