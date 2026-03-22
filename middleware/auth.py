"""API Key authentication middleware"""
import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from settings import settings

logger = logging.getLogger(__name__)

# Paths that don't require authentication
PUBLIC_PATHS = {
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware to validate API key on protected endpoints."""

    async def dispatch(self, request: Request, call_next):
        # Skip auth if no API key is configured
        if not settings.API_KEY:
            return await call_next(request)

        # Allow public paths
        path = request.url.path
        if path in PUBLIC_PATHS or path.startswith("/docs") or path.startswith("/redoc"):
            return await call_next(request)

        # Allow OPTIONS (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Check API key
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if api_key != settings.API_KEY:
            logger.warning(f"Unauthorized request to {path} from {request.client.host}")
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key. Provide via X-API-Key header."},
            )

        return await call_next(request)
