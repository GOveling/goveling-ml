"""Pydantic models for routing endpoints - replaces raw dict inputs."""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class RouteRequest(BaseModel):
    """Validated route request with coordinate bounds checking."""
    start_lat: float = Field(..., ge=-90, le=90, description="Origin latitude")
    start_lon: float = Field(..., ge=-180, le=180, description="Origin longitude")
    end_lat: float = Field(..., ge=-90, le=90, description="Destination latitude")
    end_lon: float = Field(..., ge=-180, le=180, description="Destination longitude")


class RoutePointRequest(BaseModel):
    """Route request using origin/destination objects."""
    origin: dict = Field(..., description="Origin with lat/lon")
    destination: dict = Field(..., description="Destination with lat/lon")


class CacheActionRequest(BaseModel):
    """Request for cache operations."""
    mode: Literal["drive", "walk", "bike", "all"] = Field(
        ..., description="Cache mode to act on"
    )
