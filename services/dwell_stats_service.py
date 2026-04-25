"""
DwellStatsLookup — fallback pyramid for per-place visit-duration estimates.

The optimizer needs a `duration_minutes` per POI. Historically it fell back to a
hardcoded map keyed by `place_type` (e.g. `museum=120`, `restaurant=90`). With
Phase 1 now writing closed-visit dwell rows to `trip_visits.dwell_seconds` and
Phase 2.1+2.2 aggregating those into `place_dwell_stats` / `category_dwell_stats`,
we can use real-world percentiles as a smarter mid-tier fallback.

Pyramid (highest priority first):
  1. user-supplied `duration_minutes` on the request payload (untouched here)
  2. `place_dwell_stats.p50_seconds` for the exact `place_id` (≥10 confirmed visits)
  3. `category_dwell_stats.p50_seconds` for `(category, geohash5)` (≥30 confirmed visits)
  4. hardcoded `calculate_visit_duration(type)` map in api.py

Tiers 2/3 are read from Supabase via the public-read RLS policy on the two
aggregate tables (anon key is sufficient — see migration 20260425150000).

Design notes:
- One batch fetch per optimizer request: `await build_for_places(places)`.
  Subsequent `lookup()` calls are sync dict lookups.
- Returns `None` when no stat covers the place; the caller (api.py) decides
  the final fallback. We never return the hardcoded map ourselves — that
  keeps the layering explicit.
- Graceful degradation: if `SUPABASE_URL` / `SUPABASE_ANON_KEY` are missing,
  or the HTTP request fails / times out, an empty lookup is returned and
  every `lookup()` resolves to `None`. The optimizer continues with the
  hardcoded fallback. We log at WARN once so this doesn't go silent forever.
- Geohash precision 5 mirrors the SQL `encode_geohash(lat, lng, 5)` used at
  aggregation time. Keeping the encoding identical here is critical — a
  one-bit drift produces a different cell key and silently drops every
  category-level hit.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Set, Tuple

import httpx

from settings import settings

logger = logging.getLogger(__name__)

_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def encode_geohash(lat: float, lon: float, precision: int = 5) -> Optional[str]:
    """Standard base32 geohash encoder. Mirrors `public.encode_geohash` SQL.

    Returns None when inputs are out-of-range or null. Precision 5 ≈ 4.9km
    cells; values are deterministic and bit-identical to the SQL function so
    that lookups against `category_dwell_stats.city_geohash` hit.
    """
    if lat is None or lon is None or precision <= 0:
        return None
    if lat < -90 or lat > 90 or lon < -180 or lon > 180:
        return None

    lat_min, lat_max = -90.0, 90.0
    lon_min, lon_max = -180.0, 180.0
    hash_chars: List[str] = []
    bit = 0
    ch = 0
    is_lon = True

    while len(hash_chars) < precision:
        if is_lon:
            mid = (lon_min + lon_max) / 2.0
            if lon > mid:
                ch |= 1 << (4 - bit)
                lon_min = mid
            else:
                lon_max = mid
        else:
            mid = (lat_min + lat_max) / 2.0
            if lat > mid:
                ch |= 1 << (4 - bit)
                lat_min = mid
            else:
                lat_max = mid
        is_lon = not is_lon
        if bit < 4:
            bit += 1
        else:
            hash_chars.append(_GEOHASH_BASE32[ch])
            bit = 0
            ch = 0

    return "".join(hash_chars)


class DwellStatsLookup:
    """Read-only view over learned dwell percentiles. Build via
    `await build_for_places(...)`; query via sync `lookup(...)`.
    """

    def __init__(
        self,
        place_p50: Optional[Dict[str, int]] = None,
        category_p50: Optional[Dict[Tuple[str, str], int]] = None,
    ) -> None:
        self._place_p50 = place_p50 or {}
        self._category_p50 = category_p50 or {}

    @property
    def has_data(self) -> bool:
        return bool(self._place_p50) or bool(self._category_p50)

    def lookup(
        self,
        place_id: Optional[str],
        category: Optional[str],
        lat: Optional[float],
        lon: Optional[float],
    ) -> Optional[int]:
        """Return the best available dwell-percentile in **minutes**, or None.

        Place-level hit wins over category-level hit. We use p50 because it's
        the most representative single-number summary; p25/p75 could drive
        confidence intervals later.
        """
        if place_id:
            seconds = self._place_p50.get(place_id)
            if seconds is not None:
                return max(1, round(seconds / 60))

        if category and lat is not None and lon is not None:
            geohash = encode_geohash(lat, lon, 5)
            if geohash is not None:
                seconds = self._category_p50.get((category.lower(), geohash))
                if seconds is not None:
                    return max(1, round(seconds / 60))

        return None


def _extract_place_id(place: Dict) -> Optional[str]:
    """Resolve the place identifier the dwell tables are keyed on. Mobile
    sends the Google Place ID under `id` (RTK MLPlace contract); some legacy
    paths use `place_id` or `google_place_id`. We try all three in priority
    order. Synthesized `unknown_<hash>` ids are filtered out — they would
    never match a stat row.
    """
    for key in ("id", "google_place_id", "place_id"):
        value = place.get(key)
        if value and isinstance(value, str) and not value.startswith("unknown_"):
            return value
    return None


def _extract_category(place: Dict) -> Optional[str]:
    raw = place.get("category") or place.get("type")
    if not raw:
        return None
    if hasattr(raw, "value"):  # PlaceType enum
        raw = raw.value
    return str(raw).lower() or None


def _extract_coords(place: Dict) -> Tuple[Optional[float], Optional[float]]:
    lat = place.get("lat")
    lon = place.get("lon") or place.get("long") or place.get("lng")
    try:
        return (float(lat) if lat is not None else None,
                float(lon) if lon is not None else None)
    except (TypeError, ValueError):
        return (None, None)


async def _fetch_place_p50(
    client: httpx.AsyncClient,
    base_url: str,
    headers: Dict[str, str],
    place_ids: Set[str],
) -> Dict[str, int]:
    if not place_ids:
        return {}

    # PostgREST `in.(...)` filter. Place IDs from Google can contain commas
    # in theory but in practice they're URL-safe (alphanumeric + `_-`). We
    # still wrap in double-quotes per PostgREST guidance to be safe; commas
    # inside quoted values are literal.
    quoted = ",".join(f'"{pid}"' for pid in place_ids)
    url = (
        f"{base_url}/rest/v1/place_dwell_stats"
        f"?select=place_id,p50_seconds&place_id=in.({quoted})"
    )
    resp = await client.get(url, headers=headers, timeout=5.0)
    resp.raise_for_status()
    rows = resp.json() or []
    return {
        row["place_id"]: row["p50_seconds"]
        for row in rows
        if row.get("place_id") and row.get("p50_seconds") is not None
    }


async def _fetch_category_p50(
    client: httpx.AsyncClient,
    base_url: str,
    headers: Dict[str, str],
    category_geohash_pairs: Set[Tuple[str, str]],
) -> Dict[Tuple[str, str], int]:
    if not category_geohash_pairs:
        return {}

    # category_dwell_stats uses a composite key. PostgREST doesn't directly
    # support tuple-IN filters, so we fetch all rows for the involved
    # categories and the involved geohashes (a small Cartesian over-fetch
    # bounded by len(places)) and filter client-side. For typical requests
    # (<20 places) this is at most a few dozen rows.
    categories = sorted({c for c, _ in category_geohash_pairs})
    geohashes = sorted({g for _, g in category_geohash_pairs})
    cat_filter = ",".join(f'"{c}"' for c in categories)
    geo_filter = ",".join(f'"{g}"' for g in geohashes)
    url = (
        f"{base_url}/rest/v1/category_dwell_stats"
        f"?select=category,city_geohash,p50_seconds"
        f"&category=in.({cat_filter})&city_geohash=in.({geo_filter})"
    )
    resp = await client.get(url, headers=headers, timeout=5.0)
    resp.raise_for_status()
    rows = resp.json() or []
    out: Dict[Tuple[str, str], int] = {}
    for row in rows:
        key = (row.get("category"), row.get("city_geohash"))
        p50 = row.get("p50_seconds")
        if key in category_geohash_pairs and p50 is not None:
            # Stored category may have arbitrary case from trip_places.category;
            # normalize on read so callers can pass lowercase.
            out[(key[0].lower(), key[1])] = p50
    return out


async def build_for_places(places: Iterable[Dict]) -> DwellStatsLookup:
    """One-shot batch fetch of dwell stats covering every place in the request.

    Always returns a usable lookup — failures are logged and produce an empty
    lookup so the caller can proceed with the hardcoded duration map.
    """
    base_url = (settings.SUPABASE_URL or "").rstrip("/")
    anon_key = settings.SUPABASE_ANON_KEY
    if not base_url or not anon_key:
        logger.info(
            "[DwellStatsLookup] Supabase env not configured; using empty lookup. "
            "Set SUPABASE_URL and SUPABASE_ANON_KEY to enable learned durations."
        )
        return DwellStatsLookup()

    place_ids: Set[str] = set()
    pairs: Set[Tuple[str, str]] = set()
    for place in places:
        if not isinstance(place, dict):
            continue
        pid = _extract_place_id(place)
        if pid:
            place_ids.add(pid)
        category = _extract_category(place)
        lat, lon = _extract_coords(place)
        if category and lat is not None and lon is not None:
            geohash = encode_geohash(lat, lon, 5)
            if geohash:
                pairs.add((category, geohash))

    if not place_ids and not pairs:
        return DwellStatsLookup()

    headers = {
        "apikey": anon_key,
        "Authorization": f"Bearer {anon_key}",
        "Accept": "application/json",
    }
    try:
        async with httpx.AsyncClient() as client:
            place_p50 = await _fetch_place_p50(client, base_url, headers, place_ids)
            category_p50 = await _fetch_category_p50(client, base_url, headers, pairs)
    except Exception as e:  # network / 5xx / parse — none should fail the optimizer
        logger.warning("[DwellStatsLookup] fetch failed, falling back: %s", e)
        return DwellStatsLookup()

    if place_p50 or category_p50:
        logger.info(
            "[DwellStatsLookup] Loaded %d place stats, %d category stats "
            "(of %d places, %d category-cells requested)",
            len(place_p50), len(category_p50), len(place_ids), len(pairs),
        )
    return DwellStatsLookup(place_p50=place_p50, category_p50=category_p50)
