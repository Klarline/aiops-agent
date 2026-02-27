"""Lightweight API protection for mutating endpoints."""

from __future__ import annotations

import os

from fastapi import Header, HTTPException


def require_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None),
) -> None:
    """Verify API key when AIOPS_API_KEY is set. When unset, allow all requests."""
    configured = os.environ.get("AIOPS_API_KEY")
    if not configured:
        return

    provided = x_api_key
    if not provided and authorization and authorization.startswith("Bearer "):
        provided = authorization[7:].strip()

    if not provided or provided != configured:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
