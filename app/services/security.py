from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Header, HTTPException

from app.core.config import settings

_rate_lock = threading.Lock()
_request_windows: Dict[str, Deque[float]] = defaultdict(deque)


def _parse_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2:
        return None
    scheme, token = parts
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None


def verify_api_key(authorization: str | None = Header(default=None)) -> None:
    if not settings.require_api_key:
        return
    token = _parse_bearer_token(authorization)
    if token != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def enforce_rate_limit(client_id: str) -> None:
    # Sliding-window limiter: max N requests in the last 60 seconds.
    now = time.time()
    cutoff = now - 60.0
    with _rate_lock:
        window = _request_windows[client_id]
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= settings.rate_limit_per_minute:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
        window.append(now)


def client_key_from_request(host: str | None, authorization: str | None) -> str:
    token = _parse_bearer_token(authorization)
    if token:
        return f"token:{token[:8]}"
    return f"ip:{host or 'unknown'}"
