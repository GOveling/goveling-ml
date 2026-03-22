"""City2Graph / semantic analysis endpoints."""
import logging
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException
from models.schemas import Place
from settings import settings
from utils.global_city2graph import (
    global_city2graph, get_semantic_status,
    enhance_places_with_semantic_context,
)
from utils.global_real_city2graph import (
    global_real_city2graph, get_real_semantic_status,
    enhance_places_with_real_semantic_context,
    get_global_real_semantic_clustering,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Semantic Analysis"])

# Global semantic mode
_SEMANTIC_MODE = "auto"


def _places_to_dicts(places: List[Place]) -> list:
    """Convert Place models to dicts for semantic services."""
    return [
        {
            "name": p.name,
            "lat": p.lat,
            "lon": p.lon,
            "type": p.type.value if hasattr(p.type, "value") else str(p.type),
            "rating": getattr(p, "rating", 4.5),
            "priority": getattr(p, "priority", 5),
        }
        for p in places
    ]


@router.get("/semantic/status")
async def semantic_status():
    """Estado del sistema semantico City2Graph."""
    demo = get_semantic_status()
    real = get_real_semantic_status()
    return {
        "semantic_enabled": demo["enabled"],
        "service_status": demo["service_status"],
        "features": demo["features"],
        "real_osm_enabled": real["enabled"],
        "real_service_status": real["service_status"],
        "real_features": real["features"],
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/semantic/analyze")
async def semantic_analyze_places(places: List[Place]):
    """Analisis semantico detallado (Demo)."""
    try:
        data = _places_to_dicts(places)
        enhanced = await enhance_places_with_semantic_context(data)
        from utils.global_city2graph import get_global_semantic_clustering
        clustering = await get_global_semantic_clustering(data)
        return {
            "places_analyzed": len(data),
            "enhanced_places": enhanced,
            "semantic_clustering": clustering,
            "data_source": "demo_simulated",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error en analisis semantico: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/semantic/analyze-real")
async def semantic_analyze_places_real(places: List[Place]):
    """Analisis semantico con datos OSM reales."""
    try:
        data = _places_to_dicts(places)
        enhanced = await enhance_places_with_real_semantic_context(data)
        clustering = await get_global_real_semantic_clustering(data)
        return {
            "places_analyzed": len(data),
            "enhanced_places": enhanced,
            "semantic_clustering": clustering,
            "data_source": "openstreetmap_complete",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error en analisis semantico REAL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/semantic/city/{city_name}")
async def semantic_city_summary(city_name: str):
    """Resumen semantico de una ciudad (Demo)."""
    try:
        summary = await global_city2graph.get_city_summary(city_name)
        return {
            "city": city_name,
            "city_summary": summary,
            "semantic_available": global_city2graph.is_semantic_enabled(),
            "data_source": "demo_simulated",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/semantic/city-real/{city_name}")
async def semantic_city_summary_real(city_name: str):
    """Resumen semantico con datos OSM reales."""
    try:
        summary = await global_real_city2graph.get_real_city_summary(city_name)
        return {
            "city": city_name,
            "city_summary": summary,
            "real_semantic_available": global_real_city2graph.is_real_semantic_enabled(),
            "data_source": "openstreetmap_complete",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/semantic/config")
async def set_semantic_mode(mode: str):
    """Configurar modo de analisis semantico: demo, real, o auto."""
    global _SEMANTIC_MODE
    valid = ("demo", "real", "auto")
    if mode not in valid:
        raise HTTPException(status_code=400, detail=f"Modo invalido. Opciones: {valid}")
    _SEMANTIC_MODE = mode
    return {"mode_set": mode, "timestamp": datetime.now().isoformat()}


@router.get("/semantic/config")
async def get_semantic_mode():
    """Configuracion actual del modo semantico."""
    return {
        "current_mode": _SEMANTIC_MODE,
        "demo_available": get_semantic_status()["enabled"],
        "real_available": get_real_semantic_status()["enabled"],
        "timestamp": datetime.now().isoformat(),
    }
