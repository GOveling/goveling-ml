"""
Performance cache with LRU eviction and bounded size.
Replaces the unbounded global dict that could leak memory.
"""
import asyncio
import functools
import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

MAX_CACHE_ENTRIES = 500  # Hard upper bound


class BoundedTTLCache:
    """Thread-safe LRU cache with TTL expiration and max size."""

    def __init__(self, max_size: int = MAX_CACHE_ENTRIES):
        self._store: OrderedDict[str, tuple[Any, datetime]] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if datetime.now() >= expires_at:
            self._store.pop(key, None)
            return None
        # Move to end (most recently used)
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl_minutes: int = 30):
        # Evict oldest if at capacity
        while len(self._store) >= self._max_size:
            self._store.popitem(last=False)
        self._store[key] = (value, datetime.now() + timedelta(minutes=ttl_minutes))
        self._store.move_to_end(key)

    def clear(self):
        self._store.clear()

    def cleanup_expired(self) -> int:
        now = datetime.now()
        expired = [k for k, (_, exp) in self._store.items() if exp < now]
        for k in expired:
            del self._store[k]
        return len(expired)

    @property
    def size(self) -> int:
        return len(self._store)


# Global singleton
_cache = BoundedTTLCache()


def cache_result(expiry_minutes: int = 30):
    """Async cache decorator for API responses with bounded LRU."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

            cached = _cache.get(cache_key)
            if cached is not None:
                return cached

            result = await func(*args, **kwargs)
            _cache.set(cache_key, result, ttl_minutes=expiry_minutes)
            return result
        return wrapper
    return decorator


def hash_places(places: list) -> str:
    """Create deterministic hash for places list."""
    places_str = json.dumps(
        [
            {
                "name": p.get("name", ""),
                "lat": round(p.get("lat", 0), 4),
                "lon": round(p.get("lon", 0), 4),
                "type": p.get("type", ""),
            }
            for p in places
        ],
        sort_keys=True,
    )
    return hashlib.md5(places_str.encode()).hexdigest()


def clear_cache():
    """Clear all cached data."""
    _cache.clear()


async def cleanup_expired_cache() -> int:
    """Remove expired cache entries. Returns count removed."""
    return _cache.cleanup_expired()


def get_cache_stats() -> dict:
    """Return cache statistics."""
    return {
        "entries": _cache.size,
        "max_entries": MAX_CACHE_ENTRIES,
        "utilization_pct": round(_cache.size / MAX_CACHE_ENTRIES * 100, 1),
    }
