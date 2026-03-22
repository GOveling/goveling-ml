"""
Multi-modal routing endpoints.
DRY: drive/walk/bike share a single implementation.
"""
import logging
import time as time_module
from datetime import datetime
from fastapi import APIRouter, HTTPException
from models.route_schemas import RouteRequest, CacheActionRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Multi-Modal Routing"])

# ---- lazy-loaded router singleton ----
_chile_multimodal_router = None


def get_chile_router():
    """Get or initialize the multi-modal router (lazy loading with S3 download)."""
    global _chile_multimodal_router

    if _chile_multimodal_router is None:
        try:
            from services.chile_multimodal_router import ChileMultiModalRouter

            try:
                from utils.s3_graphs_manager import S3GraphsManager
                s3_manager = S3GraphsManager()
                if s3_manager.s3_client:
                    logger.info("Verificando grafos en S3...")
                    s3_manager.ensure_critical_graphs()
            except Exception as e:
                logger.warning(f"S3 download failed: {e} - using local graphs")

            _chile_multimodal_router = ChileMultiModalRouter()
            logger.info("ChileMultiModalRouter initialized")
        except Exception as e:
            logger.error(f"Error initializing ChileMultiModalRouter: {e}")
            _chile_multimodal_router = "failed"

    return _chile_multimodal_router if _chile_multimodal_router != "failed" else None


def _require_router():
    """Get router or raise 503."""
    r = get_chile_router()
    if r is None:
        raise HTTPException(status_code=503, detail="Servicio de routing multi-modal no disponible")
    return r


# ---- DRY single-mode route ----

_MODE_LABELS = {
    "drive": "vehiculo",
    "walk": "peatonal",
    "bike": "bicicleta",
}


async def _calculate_single_route(req: RouteRequest, mode: str):
    """Shared implementation for drive/walk/bike endpoints."""
    r = _require_router()
    start = time_module.time()

    route = r.get_route(
        start_lat=req.start_lat,
        start_lon=req.start_lon,
        end_lat=req.end_lat,
        end_lon=req.end_lon,
        mode=mode,
    )

    if not route or not route.get("success"):
        raise HTTPException(
            status_code=404,
            detail=f"No se pudo calcular la ruta en {_MODE_LABELS.get(mode, mode)}",
        )

    route["performance"] = {
        "processing_time_ms": round((time_module.time() - start) * 1000, 2),
        "mode": mode,
    }
    logger.info(f"Route {mode}: {route['distance_km']}km, {route['time_minutes']}min")
    return route


@router.post("/route/drive")
async def route_drive(request: RouteRequest):
    """Calcular ruta en vehiculo."""
    return await _calculate_single_route(request, "drive")


@router.post("/route/walk")
async def route_walk(request: RouteRequest):
    """Calcular ruta peatonal."""
    return await _calculate_single_route(request, "walk")


@router.post("/route/bike")
async def route_bike(request: RouteRequest):
    """Calcular ruta en bicicleta."""
    return await _calculate_single_route(request, "bike")


@router.post("/route/compare")
async def route_compare(request: RouteRequest):
    """Comparar rutas entre todos los modos de transporte."""
    r = _require_router()
    start = time_module.time()

    routes = r.calculate_multimodal_routes(
        start_lat=request.start_lat,
        start_lon=request.start_lon,
        end_lat=request.end_lat,
        end_lon=request.end_lon,
    )

    successful_routes = {
        mode: route for mode, route in routes.items()
        if route and route.get("success")
    }
    if not successful_routes:
        raise HTTPException(status_code=404, detail="No se pudo calcular ninguna ruta")

    fastest_mode = min(successful_routes, key=lambda m: successful_routes[m]["time_minutes"])
    shortest_mode = min(successful_routes, key=lambda m: successful_routes[m]["distance_km"])

    if "walk" in successful_routes and successful_routes["walk"]["time_minutes"] <= 15:
        recommended_mode, reason = "walk", "Distancia corta - caminar es eficiente y saludable"
    elif "bike" in successful_routes and successful_routes["bike"]["time_minutes"] <= 30:
        recommended_mode, reason = "bike", "Distancia media - bicicleta es rapida y ecologica"
    else:
        recommended_mode, reason = "drive", "Distancia larga - vehiculo es la opcion mas practica"

    return {
        "routes": successful_routes,
        "analysis": {
            "fastest_mode": fastest_mode,
            "fastest_time_minutes": successful_routes[fastest_mode]["time_minutes"],
            "shortest_mode": shortest_mode,
            "shortest_distance_km": successful_routes[shortest_mode]["distance_km"],
            "recommended_mode": recommended_mode,
            "recommendation_reason": reason,
            "modes_available": list(successful_routes),
            "modes_failed": [m for m, r in routes.items() if not r or not r.get("success")],
        },
        "performance": {
            "processing_time_ms": round((time_module.time() - start) * 1000, 2),
            "routes_calculated": len(successful_routes),
            "total_modes_attempted": len(routes),
        },
        "timestamp": datetime.now().isoformat(),
    }


# ---- Health ----

@router.get("/health/multimodal")
async def multimodal_health_check():
    """Health check del sistema multi-modal."""
    try:
        start = time_module.time()
        r = get_chile_router()
        if r is None:
            return {"status": "unavailable", "message": "Router multi-modal no disponible", "timestamp": datetime.now().isoformat()}

        cache_status = r.get_cache_status()
        memory_usage = r.get_memory_usage()
        performance_stats = r.get_performance_stats()

        modes_available = [m for m, c in cache_status.items() if c["exists"]]
        modes_in_memory = sum(c.get("loaded_in_memory", False) for c in cache_status.values())

        availability_score = (len(modes_available) / 3) * 50
        total_requests = performance_stats["performance_summary"]["total_requests"]
        if total_requests > 0:
            efficiency_score = performance_stats["performance_summary"]["overall_hit_ratio"] * 50
        else:
            efficiency_score = 50
        health_score = availability_score + efficiency_score

        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 50:
            status = "degraded"
        else:
            status = "critical"

        return {
            "status": status,
            "health_score": round(health_score, 1),
            "modes_available": modes_available,
            "modes_in_memory": modes_in_memory,
            "total_modes": 3,
            "cache_status": cache_status,
            "memory_usage": memory_usage,
            "performance_stats": performance_stats["performance_summary"],
            "total_cache_size_mb": round(sum(c["size"] for c in cache_status.values() if c["exists"]), 2),
            "performance": {"health_check_time_ms": round((time_module.time() - start) * 1000, 2)},
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error en health check multi-modal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- Cache management ----

@router.post("/cache/preload")
async def preload_cache(request: CacheActionRequest):
    """Pre-cargar cache especifico para optimizacion."""
    r = _require_router()
    start = time_module.time()

    if request.mode == "all":
        results = r.preload_all_caches()
        return {
            "success": True,
            "mode": "all",
            "results": results,
            "successful_loads": sum(results.values()),
            "total_modes": len(results),
            "processing_time_ms": round((time_module.time() - start) * 1000, 2),
            "timestamp": datetime.now().isoformat(),
        }

    success = r.preload_cache(request.mode)
    return {
        "success": success,
        "mode": request.mode,
        "processing_time_ms": round((time_module.time() - start) * 1000, 2),
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/cache/clear")
async def clear_routing_cache(request: CacheActionRequest):
    """Limpiar cache de memoria para liberar RAM."""
    r = _require_router()
    memory_before = r.get_memory_usage()

    if request.mode == "all":
        r.clear_memory_cache()
    else:
        r.clear_memory_cache(request.mode)

    memory_after = r.get_memory_usage()
    return {
        "success": True,
        "mode": request.mode,
        "memory_freed_mb": round(memory_before["total_estimated_mb"] - memory_after["total_estimated_mb"], 2),
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/cache/optimize")
async def optimize_memory():
    """Optimizar uso de memoria basado en patrones de uso."""
    r = _require_router()
    return {"success": True, "optimization_report": r.optimize_memory(), "timestamp": datetime.now().isoformat()}


@router.get("/performance/stats")
async def get_performance_statistics():
    """Estadisticas detalladas de rendimiento del sistema multi-modal."""
    r = _require_router()
    stats = r.get_performance_stats()
    stats["generated_at"] = datetime.now().isoformat()
    return stats
