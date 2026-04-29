# api.py
import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import asyncio
from datetime import datetime, time as dt_time, timedelta
import time as time_module

from models.schemas import (
    Place, PlaceType, TransportMode, Coordinates,
    ItineraryRequest, ItineraryResponse, HotelRecommendationRequest,
    Activity, MultiCityOptimizationRequest, MultiCityItineraryResponse,
    MultiCityAnalysisRequest, MultiCityAnalysisResponse,
)
from settings import settings
from services.hotel_recommender import HotelRecommender
from services.google_places_service import GooglePlacesService
from services.multi_city_optimizer_simple import MultiCityOptimizerSimple
from services.city_clustering_service import CityClusteringService
from services.dwell_stats_service import build_for_places as build_dwell_stats_lookup
from utils.logging_config import setup_production_logging
from utils.performance_cache import cache_result, hash_places, cleanup_expired_cache
from utils.hybrid_optimizer_v31 import HybridOptimizerV31
from utils.global_city2graph import global_city2graph, get_semantic_status, enhance_places_with_semantic_context
from utils.global_real_city2graph import global_real_city2graph, get_real_semantic_status, enhance_places_with_real_semantic_context, get_global_real_semantic_clustering
from services.hybrid_city2graph_service import get_hybrid_service
from utils.geo_utils import haversine_km
from services.ortools_monitoring import ortools_monitor, get_monitoring_dashboard, get_benchmark_report

# Routers
from routers.routing import router as routing_router, get_chile_router
from models.route_schemas import RoutePointRequest
from routers.monitoring import router as monitoring_router
from routers.semantic import router as semantic_router

# Middleware
from middleware.auth import APIKeyMiddleware
from middleware.rate_limit import RateLimitMiddleware

# Configurar logging optimizado
logger = setup_production_logging()

# Servicio híbrido global (se inicializa al startup)
hybrid_routing_service = None

def calculate_real_route(origin_lat: float, origin_lon: float, 
                        dest_lat: float, dest_lon: float) -> Dict:
    """Calcula ruta real usando el servicio híbrido, con fallback a haversine"""
    
    # Fallback a haversine si el servicio no está disponible
    def haversine_fallback():
        distance_km = haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
        # Estimación simple de tiempo basada en distancia
        estimated_speed = 50  # km/h promedio
        travel_time_minutes = (distance_km / estimated_speed) * 60
        return {
            "distance_km": distance_km,
            "travel_time_minutes": travel_time_minutes,
            "method": "haversine_fallback",
            "estimated_speed_kmh": estimated_speed
        }
    
    service = get_or_initialize_hybrid_service()
    
    if service is None:
        logger.debug("🔄 Usando fallback haversine (servicio híbrido no disponible)")
        return haversine_fallback()
    
    try:
        result = service.route(origin_lat, origin_lon, dest_lat, dest_lon)
        
        if result:
            return {
                "distance_km": result.distance_m / 1000,
                "travel_time_minutes": result.travel_time_s / 60,
                "method": "hybrid_routing",
                "estimated_speed_kmh": result.estimated_speed_kmh,
                "highway_types": list(set(result.highway_types))
            }
        else:
            logger.debug("🔄 Ruta híbrida falló, usando fallback haversine")
            return haversine_fallback()
            
    except Exception as e:
        logger.debug(f"🔄 Error en routing híbrido: {e}, usando fallback haversine")
        return haversine_fallback()

# ========================================================================
# 🧠 CITY2GRAPH DECISION ALGORITHM - FASE 1 (NO AFECTA ENDPOINTS ACTUALES)
# ========================================================================

async def should_use_city2graph(request: ItineraryRequest) -> Dict[str, Any]:
    """
    🧠 Algoritmo inteligente para decidir qué optimizador usar
    
    Analiza la complejidad del request y determina si City2Graph agregaría valor
    vs. usar el sistema clásico (más rápido y confiable).
    
    Returns:
        Dict con decisión, score de complejidad, factores y reasoning
    """
    from utils.geo_utils import haversine_km
    
    # 🔴 Validaciones de seguridad - Master switches
    if not settings.ENABLE_CITY2GRAPH:
        return {
            "use_city2graph": False, 
            "reason": "city2graph_disabled",
            "complexity_score": 0.0,
            "factors": {}
        }
    
    # 📊 Calcular factores de complejidad
    complexity_factors = {}
    
    # Factor 1: Cantidad de lugares (peso: 3)
    places_count = len(request.places)
    complexity_factors["places_complexity"] = {
        "value": places_count,
        "score": min(places_count / settings.CITY2GRAPH_MIN_PLACES, 2.0) * 3,
        "threshold": settings.CITY2GRAPH_MIN_PLACES,
        "description": f"{places_count} lugares ({'complejo' if places_count >= settings.CITY2GRAPH_MIN_PLACES else 'simple'})"
    }
    
    # Factor 2: Duración del viaje (peso: 3)  
    trip_days = (request.end_date - request.start_date).days + 1  # +1 para incluir día final
    complexity_factors["duration_complexity"] = {
        "value": trip_days,
        "score": min(trip_days / settings.CITY2GRAPH_MIN_DAYS, 2.0) * 3,
        "threshold": settings.CITY2GRAPH_MIN_DAYS,
        "description": f"{trip_days} días ({'largo' if trip_days >= settings.CITY2GRAPH_MIN_DAYS else 'corto'})"
    }
    
    # Factor 3: Multi-ciudad detection (peso: 2)
    cities_detected = await _detect_multiple_cities_from_places(request.places)
    complexity_factors["multi_city"] = {
        "cities": cities_detected,
        "score": 2.0 if len(cities_detected) > 1 else 0.0,
        "description": f"{len(cities_detected)} ciudades detectadas: {', '.join(cities_detected) if cities_detected else 'ninguna'}"
    }
    
    # Factor 4: Tipos de lugares semánticos (peso: 1)
    semantic_types = _count_semantic_place_types(request.places)
    complexity_factors["semantic_richness"] = {
        "semantic_types": semantic_types,
        "score": min(len(semantic_types) / settings.CITY2GRAPH_SEMANTIC_TYPES_THRESHOLD, 1.0) * 1.0,
        "description": f"{len(semantic_types)} tipos semánticos: {', '.join(semantic_types) if semantic_types else 'ninguno'}"
    }
    
    # Factor 5: Distribución geográfica (peso: 1)
    geo_spread_km = _calculate_geographic_spread(request.places)
    complexity_factors["geographic_spread"] = {
        "spread_km": geo_spread_km,
        "score": min(geo_spread_km / settings.CITY2GRAPH_GEO_SPREAD_THRESHOLD_KM, 1.0) * 1.0,
        "description": f"{geo_spread_km:.1f}km dispersión geográfica"
    }
    
    # 📊 Score total (máximo: 10)
    total_score = sum(factor["score"] for factor in complexity_factors.values())
    
    # 🎯 Decisión final
    use_city2graph = total_score >= settings.CITY2GRAPH_COMPLEXITY_THRESHOLD
    
    # 🌍 Validación por ciudades habilitadas
    if use_city2graph and settings.CITY2GRAPH_CITIES:
        enabled_cities = [city.strip().lower() for city in settings.CITY2GRAPH_CITIES.split(",") if city.strip()]
        if enabled_cities:
            cities_in_enabled = [city for city in cities_detected if city.lower() in enabled_cities]
            if not cities_in_enabled:
                use_city2graph = False
                complexity_factors["city_restriction"] = {
                    "enabled_cities": enabled_cities,
                    "detected_cities": cities_detected,
                    "description": "Ciudades detectadas no están en lista habilitada"
                }
    
    return {
        "use_city2graph": use_city2graph,
        "complexity_score": round(total_score, 2),
        "factors": complexity_factors,
        "reasoning": _generate_decision_reasoning(complexity_factors, total_score, use_city2graph),
        "timestamp": datetime.now().isoformat()
    }

def _count_semantic_place_types(places: List[Dict]) -> List[str]:
    """Contar tipos de lugares semánticamente ricos que se benefician de City2Graph"""
    semantic_types = set()
    
    for place in places:
        # Extraer type del place (puede ser enum o string)
        place_type = ""
        if hasattr(place, 'type'):
            if hasattr(place.type, 'value'):
                place_type = place.type.value  # Enum
            else:
                place_type = str(place.type)   # String
        elif isinstance(place, dict) and 'type' in place:
            place_type = str(place['type'])
        
        place_type = place_type.lower()
        
        # Lugares que se benefician de análisis semántico City2Graph
        if place_type in [
            "museum", "tourist_attraction", "park", "art_gallery",
            "church", "synagogue", "mosque", "cemetery", "natural_feature",
            "university", "library", "town_hall", "courthouse",
            "locality", "neighborhood", "sublocality", "administrative_area",
            "cultural_center", "historical_site", "monument"
        ]:
            semantic_types.add(place_type)
    
    return list(semantic_types)

async def _detect_multiple_cities_from_places(places: List[Dict]) -> List[str]:
    """Detectar si el itinerario cruza múltiples ciudades usando clustering geográfico"""
    from utils.ortools_decision_engine import ORToolsDecisionEngine
    
    try:
        # Usar clustering automático de ORTools
        decision_engine = ORToolsDecisionEngine()
        clusters = decision_engine._detect_geographic_clusters(places)
        
        if len(clusters) == 0:
            return []
        elif len(clusters) == 1:
            # Una sola ciudad/área
            cluster = clusters[0]
            return [f"cluster_{cluster['center_lat']:.1f}_{cluster['center_lon']:.1f}"]
        else:
            # Múltiples clusters = múltiples ciudades
            cities = []
            for i, cluster in enumerate(clusters):
                cities.append(f"city_{i+1}_{cluster['places_count']}_places")
            
            logger.info(f"🏙️ Detectados {len(clusters)} clusters geográficos: {cities}")
            return cities
            
    except Exception as e:
        logger.warning(f"⚠️ Error en clustering automático, usando método legacy: {e}")
        
        # Fallback al método anterior
        cities = set()
        for place in places:
            city = await _extract_city_from_place(place)
            if city:
                cities.add(city.lower())
        
        return list(cities)

async def _extract_city_from_place(place: Dict) -> Optional[str]:
    """Extraer ciudad de un lugar usando reverse geocoding automático"""
    
    # Extraer coordenadas del lugar
    lat = lon = None
    
    # Método 1: Coordenadas directas
    if isinstance(place, dict):
        lat = place.get('lat')
        lon = place.get('lon')
        if lat is None:
            lat = place.get('latitude')
        if lon is None:
            lon = place.get('longitude')
    else:
        lat = getattr(place, 'lat', None)
        lon = getattr(place, 'lon', None)
    
    if lat is None or lon is None:
        logger.debug(f"⚠️ No se pudieron extraer coordenadas del lugar")
        return None
    
    try:
        # Método 2a: Si viene de Google Places, usar Place ID para detalles completos
        google_place_id = None
        if isinstance(place, dict):
            google_place_id = place.get('google_place_id') or place.get('place_id')
        else:
            google_place_id = getattr(place, 'google_place_id', None) or getattr(place, 'place_id', None)
        
        if google_place_id and google_place_id.startswith('ChIJ'):  # Google Place IDs start with ChIJ
            from utils.google_maps_client import GoogleMapsClient
            
            client = GoogleMapsClient()
            place_details = await client.get_place_details_by_id(google_place_id)
            
            if place_details and place_details.get('address_components'):
                # Extraer ciudad de los componentes de dirección
                for component in place_details['address_components']:
                    types = component.get('types', [])
                    if 'locality' in types:
                        detected_city = component['long_name'].lower()
                        logger.info(f"🏙️ Ciudad detectada desde Google Place ID: {detected_city}")
                        return detected_city
                    elif 'administrative_area_level_2' in types:  # Fallback
                        detected_city = component['long_name'].lower()
                        logger.info(f"🏛️ Área administrativa detectada desde Google Place ID: {detected_city}")
                        return detected_city
        
        # Método 2b: Reverse geocoding con coordenadas como fallback
        from utils.google_maps_client import GoogleMapsClient
        
        client = GoogleMapsClient()
        city_info = await client.reverse_geocode_city(float(lat), float(lon))
        
        if city_info and city_info.get('city'):
            detected_city = city_info['city'].lower()
            logger.info(f"🌍 Ciudad detectada por reverse geocoding: {detected_city} ({lat:.4f}, {lon:.4f})")
            return detected_city
        
    except Exception as e:
        logger.debug(f"⚠️ Error en detección con Google Places: {e}")
    
    # Método 3: Fallback por dirección si está disponible
    address = ""
    if hasattr(place, 'address'):
        address = place.address.lower()
    elif isinstance(place, dict) and 'address' in place:
        address = place['address'].lower()
    
    if address:
        # Extraer ciudad de la dirección usando patrones comunes
        import re
        
        # Patrón: "... Ciudad, País" o "... Ciudad ..."
        city_patterns = [
            r'\b([a-záéíóúñü\s]+),\s*([a-záéíóúñü]+)$',  # "Barcelona, España"
            r'\b(\w+)\s+\d{5}',  # "Paris 75001"
            r'\b(\w+),\s*\w+$',  # "Orlando, FL"
        ]
        
        for pattern in city_patterns:
            match = re.search(pattern, address, re.IGNORECASE)
            if match:
                potential_city = match.group(1).strip().lower()
                # Validar que no sea un número de calle
                if not potential_city.isdigit() and len(potential_city) > 2:
                    logger.info(f"🏠 Ciudad extraída de dirección: {potential_city}")
                    return potential_city
    
    # Método 4: Como último recurso, extraer del nombre del lugar
    place_name = ""
    if hasattr(place, 'name'):
        place_name = place.name
    elif isinstance(place, dict) and 'name' in place:
        place_name = place['name']
    
    if place_name:
        # Buscar patrones como "Torre Eiffel, París" o "Sagrada Familia Barcelona"
        import re
        
        name_patterns = [
            r'[,\s]+([a-záéíóúñü\s]{3,})[,\s]*$',  # Después de coma o espacios
            r'\b([a-záéíóúñü]{4,})\s*$',  # Última palabra si es larga
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, place_name.lower())
            if match:
                potential_city = match.group(1).strip()
                # Evitar palabras comunes que no son ciudades
                excluded_words = ['hotel', 'restaurant', 'museum', 'park', 'center', 'centro', 'tower', 'torre']
                if potential_city not in excluded_words and len(potential_city) > 3:
                    logger.info(f"📍 Ciudad extraída del nombre: {potential_city}")
                    return potential_city
    
    logger.debug(f"❓ No se pudo detectar ciudad para lugar en ({lat:.4f}, {lon:.4f})")
    return None

def _calculate_geographic_spread(places: List[Dict]) -> float:
    """Calcular dispersión geográfica máxima en km"""
    from utils.geo_utils import haversine_km
    
    if len(places) < 2:
        return 0.0
    
    coordinates = []
    for place in places:
        # Extraer coordenadas del place
        lat, lon = None, None
        
        # Método 1: Coordenadas directas (formato del frontend)
        if isinstance(place, dict):
            lat = place.get('lat', place.get('latitude'))
            lon = place.get('lon', place.get('longitude'))
        
        # Método 2: Atributos del objeto
        if lat is None and hasattr(place, 'lat'):
            lat = place.lat
        if lon is None and hasattr(place, 'lon'):
            lon = place.lon
            
        # Método 3: Formato coordinates anidado
        if lat is None or lon is None:
            if hasattr(place, 'coordinates'):
                if hasattr(place.coordinates, 'latitude'):
                    lat, lon = place.coordinates.latitude, place.coordinates.longitude
                elif isinstance(place.coordinates, dict):
                    lat, lon = place.coordinates.get('latitude'), place.coordinates.get('longitude')
            elif isinstance(place, dict) and 'coordinates' in place:
                coords = place['coordinates']
                if isinstance(coords, dict):
                    lat, lon = coords.get('latitude'), coords.get('longitude')
        
        if lat is not None and lon is not None:
            try:
                coordinates.append((float(lat), float(lon)))
            except (ValueError, TypeError):
                continue
    
    if len(coordinates) < 2:
        return 0.0
    
    # Calcular distancia máxima entre cualquier par de lugares
    max_distance = 0.0
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            distance = haversine_km(
                coordinates[i][0], coordinates[i][1],
                coordinates[j][0], coordinates[j][1]
            )
            max_distance = max(max_distance, distance)
    
    return max_distance

def _generate_decision_reasoning(factors: Dict, total_score: float, use_city2graph: bool) -> str:
    """Generar explicación human-readable de la decisión"""
    
    reasoning_parts = []
    
    # Factores principales que influyen
    high_impact_factors = []
    for factor_name, factor_data in factors.items():
        if factor_data.get("score", 0) >= 1.0:
            high_impact_factors.append(factor_data.get("description", factor_name))
    
    if high_impact_factors:
        reasoning_parts.append(f"Factores de complejidad: {'; '.join(high_impact_factors)}")
    
    # Decisión y justificación
    threshold = settings.CITY2GRAPH_COMPLEXITY_THRESHOLD
    if use_city2graph:
        reasoning_parts.append(f"Score {total_score:.1f} ≥ {threshold} → City2Graph recomendado para análisis profundo")
    else:
        reasoning_parts.append(f"Score {total_score:.1f} < {threshold} → Sistema clásico óptimo (rápido y confiable)")
    
    return ". ".join(reasoning_parts)

# ========================================================================

app = FastAPI(
    title="Goveling ML API",
    description="API de optimización de itinerarios con ML v2.2 - Con soporte para hoteles",
    version="2.2.0",
)

# ---- CORS ----
# NOTE: allow_origins=["*"] with allow_credentials=True is invalid per the
# CORS spec and browsers will reject the response.  When credentials are
# needed, list explicit origins via the CORS_ORIGINS env var
# (comma-separated).  Otherwise we default to permissive "*" WITHOUT
# credentials, which is valid and works for public APIs.
_cors_origins_raw = os.getenv("CORS_ORIGINS", "*")
if _cors_origins_raw == "*":
    _cors_origins = ["*"]
    _cors_credentials = False
else:
    _cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
    _cors_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# ---- Security & rate-limit middleware ----
app.add_middleware(APIKeyMiddleware)
app.add_middleware(RateLimitMiddleware)

# ---- Include routers ----
app.include_router(routing_router)
app.include_router(monitoring_router)
app.include_router(semantic_router)

@app.get("/health")
async def health_check():
    """Health check básico"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.2.0"
    }

# ========================================================================
# 🧠 CITY2GRAPH TESTING ENDPOINTS - FASE 1 (PARA VALIDACIÓN)
# ========================================================================

@app.get("/city2graph/config")
async def get_city2graph_config():
    """Obtener configuración actual de City2Graph (para debugging)"""
    return {
        "enabled": settings.ENABLE_CITY2GRAPH,
        "min_places": settings.CITY2GRAPH_MIN_PLACES,
        "min_days": settings.CITY2GRAPH_MIN_DAYS,
        "complexity_threshold": settings.CITY2GRAPH_COMPLEXITY_THRESHOLD,
        "enabled_cities": settings.CITY2GRAPH_CITIES.split(",") if settings.CITY2GRAPH_CITIES else [],
        "timeout_s": settings.CITY2GRAPH_TIMEOUT_S,
        "fallback_enabled": settings.CITY2GRAPH_FALLBACK_ENABLED,
        "user_percentage": settings.CITY2GRAPH_USER_PERCENTAGE,
        "track_decisions": settings.CITY2GRAPH_TRACK_DECISIONS
    }

@app.post("/city2graph/test-decision")
async def test_city2graph_decision(request: ItineraryRequest):
    """
    🧪 Testing endpoint para probar algoritmo de decisión City2Graph
    
    NO AFECTA el sistema productivo - solo retorna qué decisión tomaría
    """
    try:
        decision = await should_use_city2graph(request)
        
        # Log para debugging si está habilitado
        if settings.DEBUG:
            logger.info(f"🧪 Test decisión City2Graph: {decision['use_city2graph']} (score: {decision['complexity_score']})")
        
        return {
            "status": "success",
            "decision": decision,
            "request_summary": {
                "places_count": len(request.places),
                "trip_days": (request.end_date - request.start_date).days + 1,
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat()
            },
            "note": "Esta es solo una simulación - no afecta el sistema productivo"
        }
        
    except Exception as e:
        logger.error(f"❌ Error en test de decisión: {e}")
        raise HTTPException(status_code=500, detail=f"Error en algoritmo de decisión: {str(e)}")

@app.get("/city2graph/stats")
async def get_city2graph_stats():
    """Estadísticas de uso de City2Graph (placeholder para métricas futuras)"""
    return {
        "status": "phase_2",
        "message": "Dual optimizer architecture implementada con Circuit Breaker",
        "current_config": {
            "enabled": settings.ENABLE_CITY2GRAPH,
            "cities_enabled": settings.CITY2GRAPH_CITIES,
            "complexity_threshold": settings.CITY2GRAPH_COMPLEXITY_THRESHOLD,
            "circuit_breaker_enabled": True
        },
        "next_phase": "Integration Testing & Performance Benchmarks"
    }

@app.get("/city2graph/circuit-breaker")
async def get_circuit_breaker_status():
    """
    🔌 Endpoint para monitorear estado del Circuit Breaker de City2Graph
    
    Útil para debugging, monitoring y dashboards de operación
    """
    try:
        # Importar función del optimizador
        from utils.hybrid_optimizer_v31 import get_circuit_breaker_status
        
        status = get_circuit_breaker_status()
        
        # Calcular tiempo desde último fallo si existe
        import time
        time_since_failure = None
        if status.get("last_failure_time"):
            time_since_failure = time.time() - status["last_failure_time"]
        
        return {
            "circuit_breaker_status": status,
            "time_since_last_failure_s": time_since_failure,
            "is_healthy": status.get("state") == "CLOSED",
            "config": {
                "failure_threshold": settings.CITY2GRAPH_FAILURE_THRESHOLD,
                "recovery_timeout": settings.CITY2GRAPH_RECOVERY_TIMEOUT,
                "timeout_s": settings.CITY2GRAPH_TIMEOUT_S,
                "fallback_enabled": settings.CITY2GRAPH_FALLBACK_ENABLED
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except ImportError as e:
        return {
            "error": "circuit_breaker_not_available",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": "circuit_breaker_error", 
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ========================================================================

@app.on_event("startup")
async def startup_event():
    """Inicializar servicios básicos al startup de la API"""
    global hybrid_routing_service

    # Validate critical settings at startup — fail fast
    warnings = []
    if not settings.GOOGLE_MAPS_API_KEY and not os.getenv("GOOGLE_MAPS_API_KEY"):
        warnings.append("GOOGLE_MAPS_API_KEY not set — routing features will use fallbacks")
    if settings.API_KEY:
        logger.info("API key authentication enabled")
    else:
        warnings.append("API_KEY not set — all endpoints are unauthenticated")

    for w in warnings:
        logger.warning(f"STARTUP WARNING: {w}")

    logger.info("API iniciada - Servicio hibrido se cargara on-demand")
    hybrid_routing_service = None

    # Schedule periodic cache cleanup every 10 minutes
    async def _cache_cleanup_loop():
        while True:
            await asyncio.sleep(600)
            try:
                removed = await cleanup_expired_cache()
                if removed:
                    logger.info(f"Cache cleanup: removed {removed} expired entries")
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

    asyncio.create_task(_cache_cleanup_loop())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    global hybrid_routing_service
    logger.info("Shutting down — cleaning up resources")
    hybrid_routing_service = None
    from utils.performance_cache import clear_cache
    clear_cache()

def get_or_initialize_hybrid_service():
    """Obtiene o inicializa el servicio híbrido (lazy loading)"""
    global hybrid_routing_service
    
    if hybrid_routing_service is None:
        try:
            logger.info("� Inicializando servicio híbrido (primera consulta)...")
            hybrid_routing_service = get_hybrid_service()
            logger.info("✅ Servicio híbrido inicializado correctamente")
        except Exception as e:
            logger.error(f"❌ Error inicializando servicio híbrido: {e}")
            hybrid_routing_service = "failed"  # Marcar como fallado
    
    return hybrid_routing_service if hybrid_routing_service != "failed" else None

@app.get("/routing/status")
async def routing_status():
    """Estado del servicio de routing híbrido"""
    service = get_or_initialize_hybrid_service()
    
    if service is None:
        return {
            "status": "not_ready",
            "message": "Servicio híbrido no disponible o falló al inicializar"
        }
    
    stats = service.get_stats()
    return {
        "status": "ready", 
        "service": "hybrid_city2graph",
        "stats": stats,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/route")
async def calculate_route(request: RoutePointRequest):
    """Calcular ruta entre dos puntos usando el servicio hibrido"""
    service = get_or_initialize_hybrid_service()

    if service is None:
        raise HTTPException(status_code=503, detail="Servicio de routing no disponible")

    try:
        origin_lat = request.origin["lat"]
        origin_lon = request.origin["lon"]
        dest_lat = request.destination["lat"]
        dest_lon = request.destination["lon"]
        
        result = service.route(origin_lat, origin_lon, dest_lat, dest_lon)
        
        if not result:
            raise HTTPException(status_code=404, detail="No se encontró ruta")
        
        # Obtener coordenadas de la ruta
        coordinates = service.get_route_coordinates(result)
        
        return {
            "status": "success",
            "route": {
                "distance_km": round(result.distance_m / 1000, 2),
                "travel_time_minutes": round(result.travel_time_s / 60, 1),
                "estimated_speed_kmh": round(result.estimated_speed_kmh, 1),
                "highway_types": list(set(result.highway_types)),
                "coordinates": coordinates[:50] if len(coordinates) > 50 else coordinates  # Limitar para API
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Campo requerido faltante: {e}")
    except Exception as e:
        logger.error(f"Error calculando ruta: {e}")
        raise HTTPException(status_code=500, detail="Error interno calculando ruta")

# Semantic endpoints moved to routers/semantic.py

if settings.DEBUG:
    @app.get("/cache/stats", tags=["Cache Management"])
    async def get_cache_stats():
        """Obtener estadísticas del sistema de caché geográfico"""
        try:
            from services.google_places_service import GooglePlacesService
            places_service = GooglePlacesService()

            stats = places_service.get_cache_stats()

            return {
                "success": True,
                "cache_stats": stats,
                "recommendations": [
                    f"Hit rate actual: {stats['cache_performance']['hit_rate_percentage']}%",
                    f"Costo ahorrado: ${stats['cache_performance']['estimated_cost_saved_usd']:.3f} USD",
                    "80-90% de reducción esperada con uso continuo"
                ]
            }

        except Exception as e:
            logger.error(f"Error obteniendo stats de caché: {e}")
            return {
                "success": False,
                "error": str(e),
                "cache_stats": None
            }

    @app.post("/cache/clear", tags=["Cache Management"])
    async def clear_cache(older_than_hours: float = 24.0):
        """Limpiar caché manualmente"""
        try:
            from utils.geographic_cache_manager import get_cache_manager
            cache_manager = get_cache_manager()

            cleared_count = cache_manager.clear_cache(older_than_hours=older_than_hours)

            return {
                "success": True,
                "cleared_entries": cleared_count,
                "message": f"Limpiadas {cleared_count} entradas > {older_than_hours}h"
            }

        except Exception as e:
            logger.error(f"Error limpiando caché: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    @app.get("/debug/suggestions")
    async def debug_suggestions(lat: float = -23.6521, lon: float = -70.3958, day: int = 1):
        """Debug endpoint para probar sugerencias"""
        try:
            from services.google_places_service import GooglePlacesService

            places_service = GooglePlacesService()

            # Probar primero el método básico
            basic_suggestions = await places_service.search_nearby(
                lat=lat,
                lon=lon,
                types=['restaurant', 'tourist_attraction', 'museum'],
                limit=3
            )

            # Probar el método real con Google Places
            real_suggestions = await places_service.search_nearby_real(
                lat=lat,
                lon=lon,
                types=['restaurant', 'tourist_attraction', 'museum'],
                limit=3,
                day_offset=day
            )

            return {
                "location": {"lat": lat, "lon": lon},
                "day": day,
                "basic_suggestions": basic_suggestions,
                "real_suggestions": real_suggestions,
                "api_key_configured": bool(places_service.api_key),
                "settings": {
                    "radius": settings.FREE_DAY_SUGGESTIONS_RADIUS_M,
                    "limit": settings.FREE_DAY_SUGGESTIONS_LIMIT,
                    "enable_real_places": settings.ENABLE_REAL_PLACES
                }
            }

        except Exception as e:
            logger.error(f"Error in debug/suggestions: {e}", exc_info=True)
            error_response = {"error": str(e), "error_type": type(e).__name__}
            import traceback
            error_response["traceback"] = traceback.format_exc()
            return error_response

@app.post("/create_itinerary", response_model=ItineraryResponse, tags=["Core"])
@app.post("/api/v1/itinerary/generate-hybrid", response_model=ItineraryResponse, tags=["Fallback"])
@app.post("/api/v2/itinerary/generate-hybrid", response_model=ItineraryResponse, tags=["Hybrid Optimizer"])
@cache_result(expiry_minutes=5)  # 5 minutos de caché
async def generate_hybrid_itinerary_endpoint(request: ItineraryRequest):
    """
    🚀 OPTIMIZADOR HÍBRIDO INTELIGENTE V3.1 ENHANCED - MÁXIMA ROBUSTEZ
    
    ✨ NUEVAS FUNCIONALIDADES V3.1:
    - � Sugerencias inteligentes para bloques libres con duración-based filtering
    - 🚶‍♂️🚗 Clasificación precisa walking vs transport (30min threshold)
    - 🛡️ Normalización robusta de campos nulos
    - 🏨 Home base inteligente (hoteles → hubs → centroide)
    - 🛤️ Actividades especiales para transfers intercity largos
    - � Recomendaciones procesables con acciones específicas
    - � Retry automático y fallbacks sintéticos
    - ⚡ Manejo de errores de API con degradación elegante
    
    📊 CARACTERÍSTICAS TÉCNICAS CORE:
    - 🗺️ Clustering geográfico automático (agrupa lugares cercanos)
    - 🏨 Clustering basado en hoteles (si se proporcionan alojamientos)
    - ⚡ Estimación híbrida de tiempos (Haversine + Google Directions API)
    - 📅 Programación multi-día inteligente con horarios realistas
    - 🎯 Optimización nearest neighbor dentro de clusters
    - 🚶‍♂️🚗🚌 Recomendaciones automáticas de transporte por tramo
    - ⏰ Respeto de horarios, buffers y tiempos de traslado
    - 💰 Eficiente en costos (solo usa Google API cuando es necesario)
    
    🛡️ ROBUSTEZ V3.1:
    - Validación estricta de entrada con normalización automática
    - Retry automático en caso de fallos temporales
    - Fallbacks sintéticos cuando APIs fallan
    - Manejo elegante de campos nulos/missing
    - Respuestas mínimas garantizadas
    
    🏨 MODO HOTELES:
    - Envía 'accommodations' con tus hoteles/alojamientos
    - Sistema agrupa lugares por proximidad a hoteles
    - Rutas optimizadas desde/hacia alojamientos
    - Información de hotel incluida en cada actividad
    
    🗺️ MODO GEOGRÁFICO:
    - No envíes 'accommodations' o envía lista vacía
    - Comportamiento actual (clustering automático)
    - Mantiene toda la funcionalidad existente
    """
    # from utils.analytics import analytics  # TODO: Implementar módulo analytics
    
    start_time = time_module.time()
    
    try:
        # 🛡️ Validación robusta de entrada
        if not request.places or len(request.places) == 0:
            raise HTTPException(
                status_code=400, 
                detail="Al menos un lugar es requerido para generar el itinerario"
            )
        
        if not request.start_date or not request.end_date:
            raise HTTPException(
                status_code=400,
                detail="Fechas de inicio y fin son requeridas"
            )
        
        # 🔧 Normalizar lugares con campos faltantes
        # Pre-fetch learned dwell stats once per request. Empty lookup if
        # Supabase isn't configured or the call fails — see dwell_stats_service.
        raw_place_dicts = []
        for place in request.places:
            if hasattr(place, 'model_dump'):
                raw_place_dicts.append(place.model_dump())
            elif hasattr(place, 'dict'):
                raw_place_dicts.append(place.dict())
            elif hasattr(place, '__dict__'):
                raw_place_dicts.append(place.__dict__)
            else:
                raw_place_dicts.append(place)
        dwell_lookup = await build_dwell_stats_lookup(raw_place_dicts)

        normalized_places = []
        for i, place in enumerate(request.places):
            try:
                place_dict = raw_place_dicts[i]
                
                # Solo log esencial en producción
                if settings.DEBUG:
                    logger.info(f"📍 Normalizando lugar {i}: {place_dict.get('name', 'sin nombre')}")
                
                # Función helper para conversión segura
                def safe_float(value, default=0.0):
                    if value is None:
                        return default
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default
                        
                def safe_int(value, default=0):
                    if value is None:
                        return default
                    try:
                        return int(value)
                    except (ValueError, TypeError):
                        return default
                        
                def safe_str(value, default=""):
                    if value is None:
                        return default
                    return str(value)
                        
                def safe_enum_value(value, default=""):
                    """Extraer el valor del enum de PlaceType"""
                    if value is None:
                        return default
                    # Si es un enum, extraer el valor
                    if hasattr(value, 'value'):
                        return value.value
                    # Si es string que contiene el formato "PlaceType.VALUE"
                    value_str = str(value)
                    if 'PlaceType.' in value_str:
                        return value_str.split('.')[-1].lower()
                    return value_str
                
                # Obtener valores de category/type con manejo de enum
                category_value = safe_enum_value(place_dict.get('category') or place_dict.get('type'), 'general')
                type_value = safe_enum_value(place_dict.get('type') or place_dict.get('category'), 'point_of_interest')
                
                # Determinar quality flag si rating < 4.5
                rating_value = max(0.0, min(5.0, safe_float(place_dict.get('rating'))))
                quality_flag = None
                if rating_value > 0 and rating_value < 4.5:
                    quality_flag = "user_provided_below_threshold"
                    logger.info(f"⚠️ Lugar con rating bajo: {place_dict.get('name', 'lugar')} ({rating_value}⭐) - marcado como user_provided")
                
                # Duration fallback pyramid:
                #   1. user-supplied duration_minutes (or min_duration_hours * 60)
                #   2. place_dwell_stats.p50 for this Google place_id
                #   3. category_dwell_stats.p50 for (category, geohash5)
                #   4. hardcoded calculate_visit_duration map
                lat_value = safe_float(place_dict.get('lat'))
                lon_value = safe_float(place_dict.get('lon'))
                user_duration = place_dict.get('duration_minutes')
                if user_duration is None and place_dict.get('min_duration_hours') is not None:
                    user_duration = int(round(float(place_dict['min_duration_hours']) * 60))
                if user_duration is not None:
                    duration_minutes = max(1, int(user_duration))
                else:
                    learned = dwell_lookup.lookup(
                        place_id=(place_dict.get('id')
                                  or place_dict.get('google_place_id')
                                  or place_dict.get('place_id')),
                        category=category_value,
                        lat=lat_value,
                        lon=lon_value,
                    )
                    duration_minutes = learned if learned is not None \
                        else calculate_visit_duration(type_value)

                normalized_place = {
                    'place_id': place_dict.get('place_id') or place_dict.get('id') or f"place_{i}",
                    'name': safe_str(place_dict.get('name'), f"Lugar {i+1}"),
                    'lat': lat_value,
                    'lon': lon_value,
                    'category': category_value,
                    'type': type_value,
                    'rating': rating_value,
                    'price_level': max(0, min(4, safe_int(place_dict.get('price_level')))),
                    'address': safe_str(place_dict.get('address')),
                    'description': safe_str(place_dict.get('description'), f"Visita a {place_dict.get('name', 'lugar')}"),
                    'photos': place_dict.get('photos') or [],
                    'opening_hours': place_dict.get('opening_hours') or {},
                    'website': safe_str(place_dict.get('website')),
                    'phone': safe_str(place_dict.get('phone')),
                    'priority': max(1, min(10, safe_int(place_dict.get('priority'), 5))),
                    'duration_minutes': duration_minutes,
                    'quality_flag': quality_flag  # Agregar quality flag
                }
                
                logger.info(f"✅ Lugar normalizado: {normalized_place['name']} ({normalized_place['lat']}, {normalized_place['lon']})")
                normalized_places.append(normalized_place)
            except Exception as e:
                logger.error(f"❌ Error normalizando lugar {i}: {e}")
                logger.error(f"   Tipo de objeto: {type(place)}")
                logger.error(f"   Contenido: {place}")
                # Skip places that fail normalization instead of inserting at (0,0)
                # which would distort the entire itinerary routing
                logger.warning(f"   ⚠️ Lugar {i} descartado por error de normalización")
        
        # 🏨 DETECCIÓN Y RECOMENDACIÓN AUTOMÁTICA DE ACCOMMODATIONS
        # 1. Detectar si hay accommodations en places ORIGINALES
        accommodations_in_places = [
            place for place in normalized_places 
            if place.get('type', '').lower() == 'accommodation' or place.get('place_type', '').lower() == 'accommodation'
        ]
        
        # 2. Flag para indicar si NO había accommodations originalmente
        no_original_accommodations = len(accommodations_in_places) == 0 and not request.accommodations
        
        # 3. Si no hay accommodations, recomendar automáticamente
        if no_original_accommodations:
            logger.info("🤖 No se encontraron accommodations, recomendando hotel automáticamente...")
            logger.info(f"📋 Lugares antes de hotel: {[p.get('name', 'Sin nombre') for p in normalized_places]}")
            try:
                from services.hotel_recommender import HotelRecommender
                hotel_recommender = HotelRecommender()
                
                logger.info(f"📍 Buscando hoteles para {len(normalized_places)} lugares")
                
                # Recomendar el mejor hotel basado en los lugares de entrada
                recommendations = await hotel_recommender.recommend_hotels(
                    normalized_places, 
                    max_recommendations=1, 
                    price_preference="any"  # Cambiar a "any" para aceptar cualquier precio
                )
                
                logger.info(f"🏨 Hotel recommender devolvió {len(recommendations)} recomendaciones")
                
                if recommendations:
                    best_hotel = recommendations[0]
                    logger.info(f"✅ Mejor hotel encontrado: {best_hotel.name}")
                    logger.info(f"📍 Ubicación: ({best_hotel.lat:.4f}, {best_hotel.lon:.4f})")
                    
                    # Agregar el hotel recomendado a la lista de lugares
                    hotel_place = {
                        'name': best_hotel.name,
                        'lat': best_hotel.lat,
                        'lon': best_hotel.lon,
                        'type': 'accommodation',
                        'place_type': 'accommodation',
                        'rating': best_hotel.rating,
                        'address': best_hotel.address,
                        'category': 'accommodation',
                        'user_ratings_total': 100,  # Valor por defecto
                        'description': f"Hotel recomendado automáticamente: {best_hotel.name}",
                        'estimated_time': '1h',
                        'image': '',
                        'website': '',
                        'phone': '',
                        'priority': 5,
                        '_auto_recommended': True  # FLAG para identificar hoteles recomendados automáticamente
                    }
                    normalized_places.append(hotel_place)
                    logger.info(f"✅ Hotel recomendado agregado: {best_hotel.name} ({best_hotel.rating}⭐)")
                    logger.info(f"🔍 DEBUG: Hotel agregado con _auto_recommended=True y {len(normalized_places)} lugares totales")
                    logger.info(f"📋 Lugares actuales: {[p.get('name', 'Sin nombre') for p in normalized_places]}")
                else:
                    logger.warning("⚠️ No se pudo recomendar ningún hotel automáticamente")
                    
            except Exception as e:
                logger.error(f"❌ Error recomendando hotel automáticamente: {e}")
        
        # 🔍 Detectar y consolidar TODOS los hoteles/alojamientos
        accommodations_data = []
        hotels_provided = False
        
        # 1. Procesar accommodations del campo dedicado
        if request.accommodations:
            try:
                for acc in request.accommodations:
                    if hasattr(acc, 'model_dump'):
                        accommodations_data.append(acc.model_dump())
                    elif hasattr(acc, 'dict'):
                        accommodations_data.append(acc.dict())
                    else:
                        accommodations_data.append(acc)
            except Exception as e:
                logger.warning(f"Error procesando request.accommodations: {e}")
        
        # 2. Procesar accommodations que vienen en places
        if accommodations_in_places:
            logger.info(f"🏨 Encontrados {len(accommodations_in_places)} accommodations en places")
            for acc_place in accommodations_in_places:
                logger.info(f"🏨 Agregando accommodation desde places: {acc_place.get('name', 'Sin nombre')}")
                accommodations_data.append(acc_place)
        
        # 3. Verificar si tenemos accommodations del usuario
        hotels_provided = len(accommodations_data) > 0
        
        # 🧠 ANÁLISIS SEMÁNTICO AUTOMÁTICO
        semantic_enabled = global_city2graph.is_semantic_enabled()
        if semantic_enabled:
            logger.info("🧠 Enriqueciendo lugares con contexto semántico")
            try:
                normalized_places = await enhance_places_with_semantic_context(normalized_places)
                logger.info(f"✅ {len(normalized_places)} lugares enriquecidos con contexto semántico")
            except Exception as e:
                logger.warning(f"⚠️ Error en enriquecimiento semántico: {e}")
        else:
            logger.info("🔴 Sistema semántico no disponible - usando análisis geográfico básico")
        
        logger.info(f"🚀 Iniciando optimización V3.1 ENHANCED {'CON SEMÁNTICA' if semantic_enabled else 'BÁSICA'} para {len(normalized_places)} lugares")
        logger.info(f"📅 Período: {request.start_date} a {request.end_date}")
        
        # Convertir fechas
        if isinstance(request.start_date, str):
            start_date = datetime.strptime(request.start_date, '%Y-%m-%d')
        else:
            start_date = datetime.combine(request.start_date, dt_time.min)
            
        if isinstance(request.end_date, str):
            end_date = datetime.strptime(request.end_date, '%Y-%m-%d')
        else:
            end_date = datetime.combine(request.end_date, dt_time.min)
        
        # 🔄 OPTIMIZACIÓN CON RETRY AUTOMÁTICO
        from utils.hybrid_optimizer_v31 import optimize_itinerary_hybrid
        
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # analytics.track_request("hybrid_itinerary_v31", {
                #     "places_count": len(normalized_places),
                #     "hotels_provided": hotels_provided,
                #     "days_requested": (end_date - start_date).days + 1,
                #     "transport_mode": request.transport_mode,
                #     "attempt": attempt + 1
                # })
                
                # Pasar información extra al optimizer
                extra_info = {
                    'no_original_accommodations': no_original_accommodations
                }
                
                optimization_result = await optimize_itinerary_hybrid(
                    normalized_places,
                    start_date,
                    end_date,
                    request.daily_start_hour,
                    request.daily_end_hour,
                    request.transport_mode,
                    accommodations_data,
                    extra_info=extra_info
                )
                
                # 🛡️ Validar resultado antes de continuar
                if not optimization_result or 'days' not in optimization_result:
                    raise ValueError("Resultado de optimización inválido")
                
                # Éxito - salir del loop de retry
                break
                
            except Exception as e:
                last_error = e
                
                # 🔍 LOGGING EXHAUSTIVO DEL ERROR
                import traceback
                error_repr = repr(e)
                error_traceback = traceback.format_exc()
                error_code = f"OPT_ERR_{attempt + 1}_{type(e).__name__}"
                
                logger.error(f"❌ {error_code}: {error_repr}")
                logger.error(f"📊 Traceback completo:\n{error_traceback}")
                logger.error(f"🔢 Intento {attempt + 1}/{max_retries} - Lugares: {len(normalized_places)}")
                
                # Analizar tipo de error específico
                if "Geographic coherence error" in str(e):
                    logger.error("🌍 Error de coherencia geográfica detectado")
                elif "google_service" in str(e):
                    logger.error("🗺️ Error de servicio Google detectado")
                elif "DBSCAN" in str(e) or "cluster" in str(e).lower():
                    logger.error("🗂️ Error de clustering detectado")
                
                if attempt < max_retries - 1:
                    # Esperar antes del siguiente intento
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
        
        # 🚨 Si todos los intentos fallaron, crear fallback mínimo
        if last_error is not None and 'optimization_result' not in locals():
            error_reason = f"{type(last_error).__name__}: {str(last_error)}"
            logger.error(f"💥 Optimización falló después de {max_retries} intentos: {error_reason}")
            
            # Respuesta de fallback básica MEJORADA - Sin duplicados
            fallback_days = {}
            day_count = (end_date - start_date).days + 1
            places_per_day = len(normalized_places) // day_count if day_count > 0 else len(normalized_places)
            remaining_places = len(normalized_places) % day_count if day_count > 0 else 0
            
            current_place_index = 0
            
            for i in range(day_count):
                current_date = start_date + timedelta(days=i)
                date_key = current_date.strftime('%Y-%m-%d')
                
                # Distribuir lugares equitativamente
                places_this_day = places_per_day + (1 if i < remaining_places else 0)
                day_places = normalized_places[current_place_index:current_place_index + places_this_day]
                current_place_index += places_this_day
                
                fallback_days[date_key] = {
                    "day": i + 1,
                    "date": date_key,
                    "activities": day_places,
                    "transfers": [],
                    "free_blocks": [],
                    "actionable_recommendations": [
                        {
                            "type": "system_error",
                            "priority": "high",
                            "title": "Optimización temporal no disponible",
                            "description": f"Error: {error_reason}. Usando itinerario básico.",
                            "action": "retry_optimization"
                        }
                    ],
                    "base": day_places[0] if day_places else None,  # Primer lugar del día como base
                    "travel_summary": {
                        "total_travel_time_s": 0,
                        "walking_time_minutes": 0,
                        "transport_time_minutes": 0
                    }
                }
            
            return ItineraryResponse(
                itinerary=list(fallback_days.values()),
                optimization_metrics={
                    "total_places": len(normalized_places),
                    "total_days": day_count,
                    "optimization_mode": "fallback_basic",
                    "error": error_reason,  # ← Propagamos error_reason aquí
                    "fallback_active": True,
                    "efficiency_score": 0.3,
                    "processing_time_seconds": time_module.time() - start_time
                },
                recommendations=[
                    f"Sistema en modo fallback - Error: {error_reason}",
                    "Intente nuevamente en unos momentos",
                    "Contacte soporte si el problema persiste"
                ]
            )
        
        # Extraer datos del resultado de optimización
        days_data = optimization_result.get("days", [])
        optimization_metrics = optimization_result.get('optimization_metrics', {})
        clusters_info = optimization_result.get('clusters_info', {})  # ← LÍNEA AÑADIDA
        
        # 🔧 CORREGIR ALIASES EN INTERCITY TRANSFERS TEMPRANO (antes de recommendations)
        if 'intercity_transfers' in optimization_metrics and days_data:
            # Construir bases referenciales desde days_data
            temp_bases = []
            for day in days_data:
                base = day.get('base', {})
                if base:
                    temp_bases.append(base)
            
            # Corregir aliases en optimization_metrics
            corrected_transfers = []
            for transfer in optimization_metrics['intercity_transfers']:
                corrected_transfer = transfer.copy()
                
                from_lat = transfer.get('from_lat', 0)
                from_lon = transfer.get('from_lon', 0)
                to_lat = transfer.get('to_lat', 0)
                to_lon = transfer.get('to_lon', 0)
                
                for base in temp_bases:
                    base_lat = base.get('lat', 0)
                    base_lon = base.get('lon', 0)
                    
                    # Corregir FROM
                    if (abs(base_lat - from_lat) < 0.01 and abs(base_lon - from_lon) < 0.01):
                        corrected_transfer['from'] = base.get('name', transfer.get('from', ''))
                    
                    # Corregir TO
                    if (abs(base_lat - to_lat) < 0.01 and abs(base_lon - to_lon) < 0.01):
                        corrected_transfer['to'] = base.get('name', transfer.get('to', ''))
                
                corrected_transfers.append(corrected_transfer)
            
            optimization_metrics['intercity_transfers'] = corrected_transfers
        
        # Contar actividades totales
        total_activities = sum(len(day.get("activities", [])) for day in days_data)
        
        # Inicializar diccionario de sugerencias para días libres
        suggestions_by_date = {}

        
        # Determinar el modo de optimización usado
        optimization_mode = "hotel_centroid" if hotels_provided else "geographic_clustering"
        
        # 🧠 OBTENER INFORMACIÓN SEMÁNTICA GLOBAL
        semantic_info = get_semantic_status()
        
        # Formatear respuesta inteligente basada en el modo usado
        base_recommendations = []
        
        # 🧠 AÑADIR INFORMACIÓN SEMÁNTICA A RECOMENDACIONES
        if semantic_info['enabled']:
            base_recommendations.append("🧠 Análisis SEMÁNTICO activado - Clustering inteligente por contexto urbano")
            if semantic_info['features']['initialized_cities']:
                cities = ', '.join(semantic_info['features']['initialized_cities'])
                base_recommendations.append(f"🏙️ Ciudades analizadas semánticamente: {cities}")
        else:
            base_recommendations.append("🔴 Análisis semántico no disponible - usando clustering geográfico básico")
        
        if hotels_provided:
            base_recommendations.extend([
                "🏨 Itinerario optimizado con hoteles como centroides",
                f"📍 {len(accommodations_data)} hotel(es) usado(s) como base",
                "⚡ Rutas optimizadas desde/hacia alojamientos",
                "🚗 Recomendaciones de transporte por tramo"
            ])
        else:
            base_recommendations.extend([
                "Itinerario optimizado con clustering geográfico automático",
                "Agrupación inteligente por proximidad geográfica"
            ])
            
        base_recommendations.extend([
            f"Método híbrido V3.1: DBSCAN + Time Windows + ETAs reales",
            f"{total_activities} actividades distribuidas en {len(days_data)} días",
            f"Score de eficiencia: {optimization_result.get('optimization_metrics', {}).get('efficiency_score', 0.9):.1%}",
            # f"Tiempo total de viaje: {int(total_travel_minutes)} minutos", # Calculado después
            f"Estrategia de empaquetado: {clusters_info.get('packing_strategy_used', 'balanced')}"
        ])
        
        # 🚗 Añadir información sobre traslados largos detectados
        if optimization_metrics.get('long_transfers_detected', 0) > 0:
            transfer_count = optimization_metrics['long_transfers_detected']
            total_intercity_time = optimization_metrics.get('total_intercity_time_hours', 0)
            total_intercity_distance = optimization_metrics.get('total_intercity_distance_km', 0)
            
            base_recommendations.extend([
                f"🚗 {transfer_count} traslado(s) interurbano(s) detectado(s)",
                f"📏 Distancia total entre ciudades: {total_intercity_distance:.0f}km", 
                f"⏱️ Tiempo total de traslados largos: {total_intercity_time:.1f}h"
            ])
            
            # Información sobre clusters (validada)
            if clusters_info:
                base_recommendations.append(f"🏨 {clusters_info.get('total_clusters', 0)} zona(s) geográfica(s) identificada(s)")
            
            # Explicar separación de clusters
            base_recommendations.append("🗺️ Clusters separados por distancia para evitar traslados imposibles el mismo día")
            
            # Añadir detalles de cada traslado si hay pocos
            if transfer_count <= 3 and 'intercity_transfers' in optimization_metrics:
                for transfer in optimization_metrics['intercity_transfers']:
                    mode_forced = "" if transfer.get('mode') == request.transport_mode else f" (modo forzado: {transfer.get('mode')})"
                    base_recommendations.append(
                        f"  • {transfer['from']} → {transfer['to']}: "
                        f"{transfer['distance_km']:.0f}km (~{transfer['estimated_time_hours']:.1f}h){mode_forced}"
                    )
            
            # Advertencia sobre modo de transporte si se forzó cambio
            if request.transport_mode == 'walk':
                base_recommendations.append(
                    "⚠️ Modo de transporte cambiado automáticamente para traslados largos (walk → drive/transit)"
                )
            
            # Información sobre hoteles recomendados (validada)
            if clusters_info and clusters_info.get('recommended_hotels', 0) > 0:
                base_recommendations.append(
                    f"🏨 {clusters_info['recommended_hotels']} hotel(es) recomendado(s) automáticamente como base"
                )
        else:
            # 🔍 CÁLCULO DINÁMICO DE ZONAS GEOGRÁFICAS
            unique_bases = set()
            intercity_transfers = []
            
            for day in days_data:
                base = day.get('base')
                if base and base.get('name'):
                    unique_bases.add(base['name'])
                
                # Recopilar transfers intercity
                for transfer in day.get('transfers', []):
                    if transfer.get('type') == 'intercity_transfer':
                        intercity_transfers.append(f"{transfer['from']} → {transfer['to']} ({transfer.get('mode', 'drive')})")
            
            unique_clusters = len(unique_bases)
            
            if unique_clusters <= 1:
                base_recommendations.append("✅ Todos los lugares están en la misma zona geográfica")
            else:
                base_recommendations.append(f"🏨 {unique_clusters} zonas geográficas identificadas")
                if intercity_transfers:
                    base_recommendations.append(f"🚗 Transfers: {', '.join(intercity_transfers)}")
        
        # Formatear respuesta para frontend simplificada
        def get_value(activity, key, default=None):
            """Helper para obtener valor tanto de objetos como diccionarios"""
            if isinstance(activity, dict):
                return activity.get(key, default)
            else:
                return getattr(activity, key, default)
        
        def calculate_dynamic_duration(activity):
            """Calcular duración dinámica basada en tipo de actividad y distancia"""
            # Si ya tiene duration_minutes, usarlo
            if get_value(activity, 'duration_minutes', 0) > 0:
                return get_value(activity, 'duration_minutes', 0)
            
            # Para transfers, verificar si ya tiene tiempo calculado en el nombre
            activity_name = str(get_value(activity, 'name', ''))
            activity_name_lower = activity_name.lower()
            is_transfer = (get_value(activity, 'category', '') == 'transfer' or 
                          get_value(activity, 'type', '') == 'transfer' or
                          'traslado' in activity_name_lower or 'transfer' in activity_name_lower)
            
            if is_transfer:
                # Primero verificar si el nombre ya incluye duración calculada
                import re
                # Buscar patrones como "(68min)" o "(3h)" o "(2.5h)" 
                minutes_match = re.search(r'\((\d+)min\)', activity_name)
                hours_match = re.search(r'\((\d+(?:\.\d+)?)h\)', activity_name)
                
                if minutes_match:
                    calculated_minutes = int(minutes_match.group(1))
                    logger.info(f"✅ Usando duración del optimizador (min): '{activity_name}' = {calculated_minutes}min")
                    return calculated_minutes
                elif hours_match:
                    calculated_hours = float(hours_match.group(1))
                    calculated_minutes = int(calculated_hours * 60)
                    logger.info(f"✅ Usando duración del optimizador (h): '{activity_name}' = {calculated_hours}h → {calculated_minutes}min")
                    return calculated_minutes
                distance_km = get_value(activity, 'distance_km', 0)
                transport_mode = get_value(activity, 'transport_mode', request.transport_mode if hasattr(request, 'transport_mode') else 'walk')
                # Si no hay distance_km, intentar calcular con coordenadas
                if distance_km <= 0:
                    origin_lat = get_value(activity, 'origin_lat')
                    origin_lon = get_value(activity, 'origin_lon')
                    dest_lat = get_value(activity, 'lat')
                    dest_lon = get_value(activity, 'lon')
                    
                    if all([origin_lat, origin_lon, dest_lat, dest_lon]):
                        from utils.geo_utils import haversine_km  
                        distance_km = haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
                        logger.debug(f"🔧 Calculando distancia para transfer '{activity_name}': {distance_km:.2f}km")
                
                if distance_km > 0:
                    # Velocidades realistas por modo
                    speeds = {
                        'walk': 4.5,      # 4.5 km/h caminando
                        'walking': 4.5,
                        'drive': 45.0,    # 45 km/h en ciudad/carretera
                        'car': 45.0,
                        'transit': 30.0,  # 30 km/h transporte público
                        'bicycle': 15.0   # 15 km/h bicicleta
                    }
                    
                    speed_kmh = speeds.get(transport_mode, 4.5)
                    duration_minutes = (distance_km / speed_kmh) * 60
                    
                    # Buffer adicional para transfers largos
                    if distance_km > 30:  # Intercity
                        duration_minutes *= 1.2  # 20% buffer
                    else:  # Urbano
                        duration_minutes *= 1.1  # 10% buffer
                    
                    calculated_time = max(5, int(duration_minutes))
                    logger.info(f"✅ Transfer dinámico: '{activity_name}' = {distance_km:.2f}km → {calculated_time}min ({transport_mode})")
                    return calculated_time
                else:
                    # Fallback para transfers sin coordenadas: tiempo estimado por modo
                    fallback_times = {
                        'walk': 20,    # 20 min caminando urbano
                        'walking': 20,
                        'drive': 15,   # 15 min conduciendo urbano  
                        'car': 15,
                        'transit': 25, # 25 min transporte público
                        'bicycle': 12  # 12 min bicicleta
                    }
                    fallback_time = fallback_times.get(transport_mode, 20)
                    logger.warning(f"⚠️ Transfer sin coordenadas '{activity_name}' - usando fallback: {fallback_time}min")
                    return fallback_time
            
            # Valores por defecto según tipo de actividad
            activity_defaults = {
                'hotel': 30,           # Check-in/check-out
                'restaurant': 90,      # Comida
                'museum': 120,         # Visita museo
                'tourist_attraction': 90,  # Atracción turística
                'park': 60,           # Parque
                'cafe': 45,           # Café
                'shopping_mall': 120, # Compras
            }
            
            category = get_value(activity, 'category', get_value(activity, 'type', 'point_of_interest'))
            return activity_defaults.get(category, 60)  # 1 hora por defecto

        def format_time_window(activity, all_activities=None, activity_index=None):
            """Formatear ventana de tiempo para actividades, especialmente transfers"""
            start_time = get_value(activity, 'start_time', 0)
            end_time = get_value(activity, 'end_time', 0)
            
            # Para transfers, calcular tiempo basado en actividad anterior si no tiene tiempos válidos
            is_transfer = (get_value(activity, 'category', '') == 'transfer' or 
                          get_value(activity, 'type', '') == 'transfer' or
                          'traslado' in str(get_value(activity, 'name', '')).lower() or
                          'viaje' in str(get_value(activity, 'name', '')).lower() or
                          'regreso' in str(get_value(activity, 'name', '')).lower())
            
            if is_transfer and (start_time == 0 or start_time == end_time) and all_activities and activity_index is not None:
                # Usar tiempo de la actividad anterior + su duración como inicio del transfer
                if activity_index > 0:
                    prev_activity = all_activities[activity_index - 1]
                    prev_end_time = get_value(prev_activity, 'end_time', 0)
                    transfer_duration = calculate_dynamic_duration_with_context(activity, all_activities, activity_index)
                    
                    if prev_end_time > 0:
                        start_time = prev_end_time
                        end_time = start_time + transfer_duration
            
            # Si aún no hay tiempos válidos, omitir best_time
            if start_time == 0 and end_time == 0:
                return None
            
            # Validar que los tiempos estén en rango válido (0-1440 minutos = 24h)
            if start_time < 0 or start_time >= 1440 or end_time < 0 or end_time >= 1440:
                return None
            
            return f"{start_time//60:02d}:{start_time%60:02d}-{end_time//60:02d}:{end_time%60:02d}"

        def calculate_dynamic_duration_with_context(activity, all_activities, activity_index):
            """Calcular duración dinámica con acceso a actividades adyacentes para inferir coordenadas"""
            # Si ya tiene duration_minutes, usarlo
            if get_value(activity, 'duration_minutes', 0) > 0:
                return get_value(activity, 'duration_minutes', 0)
            
            # Para transfers, verificar si ya tiene tiempo calculado en el nombre
            activity_name = str(get_value(activity, 'name', ''))
            activity_name_lower = activity_name.lower()
            is_transfer = (get_value(activity, 'category', '') == 'transfer' or 
                          get_value(activity, 'type', '') == 'transfer' or
                          'traslado' in activity_name_lower or 'transfer' in activity_name_lower or
                          'viaje' in activity_name_lower or 'regreso' in activity_name_lower)
            
            if is_transfer:
                # Primero verificar si el nombre ya incluye duración calculada
                import re
                minutes_match = re.search(r'\((\d+)min\)', activity_name)
                hours_match = re.search(r'\((\d+(?:\.\d+)?)h\)', activity_name)
                
                if minutes_match:
                    calculated_minutes = int(minutes_match.group(1))
                    logger.info(f"✅ Usando duración del optimizador (min): '{activity_name}' = {calculated_minutes}min")
                    return calculated_minutes
                elif hours_match:
                    calculated_hours = float(hours_match.group(1))
                    calculated_minutes = int(calculated_hours * 60)
                    logger.info(f"✅ Usando duración del optimizador (h): '{activity_name}' = {calculated_hours}h → {calculated_minutes}min")
                    return calculated_minutes
                
                # Si no hay tiempo en el nombre, intentar inferir coordenadas de actividades adyacentes
                origin_coords = None
                dest_coords = None
                
                # Actividad anterior (origen del transfer)
                if activity_index > 0:
                    prev_activity = all_activities[activity_index - 1]
                    prev_lat = get_value(prev_activity, 'lat', 0.0)
                    prev_lon = get_value(prev_activity, 'lon', 0.0)
                    if prev_lat != 0.0 and prev_lon != 0.0:
                        origin_coords = (prev_lat, prev_lon)
                
                # Actividad siguiente (destino del transfer)
                if activity_index < len(all_activities) - 1:
                    next_activity = all_activities[activity_index + 1]
                    next_lat = get_value(next_activity, 'lat', 0.0)
                    next_lon = get_value(next_activity, 'lon', 0.0)
                    if next_lat != 0.0 and next_lon != 0.0:
                        dest_coords = (next_lat, next_lon)
                
                # Calcular distancia si tenemos ambas coordenadas
                if origin_coords and dest_coords:
                    from utils.geo_utils import haversine_km
                    distance_km = haversine_km(origin_coords[0], origin_coords[1], dest_coords[0], dest_coords[1])
                    
                    # Velocidades realistas por modo de transporte
                    transport_mode = get_value(activity, 'transport_mode', request.transport_mode if hasattr(request, 'transport_mode') else 'drive')
                    speeds = {
                        'walk': 4.5,      # 4.5 km/h caminando
                        'walking': 4.5,
                        'drive': 45.0,    # 45 km/h en ciudad/carretera
                        'car': 45.0,
                        'transit': 30.0,  # 30 km/h transporte público
                        'bicycle': 15.0   # 15 km/h bicicleta
                    }
                    
                    speed_kmh = speeds.get(transport_mode, 45.0)
                    duration_minutes = (distance_km / speed_kmh) * 60
                    
                    # Buffer adicional para transfers largos
                    if distance_km > 30:  # Intercity
                        duration_minutes *= 1.2  # 20% buffer
                    else:  # Urbano
                        duration_minutes *= 1.1  # 10% buffer
                    
                    calculated_time = max(5, int(duration_minutes))
                    logger.info(f"✅ Transfer dinámico calculado: '{activity_name}' = {distance_km:.2f}km → {calculated_time}min ({transport_mode})")
                    return calculated_time
                else:
                    # Fallback para transfers sin coordenadas
                    transport_mode = get_value(activity, 'transport_mode', request.transport_mode if hasattr(request, 'transport_mode') else 'drive')
                    fallback_times = {
                        'walk': 20,    # 20 min caminando urbano
                        'walking': 20,
                        'drive': 15,   # 15 min conduciendo urbano  
                        'car': 15,
                        'transit': 25, # 25 min transporte público
                        'bicycle': 12  # 12 min bicicleta
                    }
                    fallback_time = fallback_times.get(transport_mode, 15)
                    logger.warning(f"⚠️ Transfer sin coordenadas válidas '{activity_name}' - usando fallback: {fallback_time}min")
                    return fallback_time
            
            # Valores por defecto según tipo de actividad (no transfers)
            activity_defaults = {
                'hotel': 30,           # Check-in/check-out
                'restaurant': 90,      # Comida
                'museum': 120,         # Visita museo
                'tourist_attraction': 90,  # Atracción turística
                'park': 60,           # Parque
                'cafe': 45,           # Café
                'shopping_mall': 120, # Compras
            }
            
            category = get_value(activity, 'category', get_value(activity, 'type', 'point_of_interest'))
            return activity_defaults.get(category, 60)  # 1 hora por defecto

        def format_activity_for_frontend(activity, order, all_activities=None, activity_index=None):
            """Convertir ActivityItem o IntercityActivity a formato esperado por frontend"""
            import uuid
            
            # Detectar si es una actividad intercity
            is_intercity = get_value(activity, 'is_intercity_activity', False) or get_value(activity, 'type', '') == 'intercity_activity'
            
            if is_intercity:
                # Las actividades intercity NO deben aparecer como places individuales
                # Solo aparecen en optimization_metrics.intercity_transfers
                return None
            else:
                # Detectar si es un transfer para agregar coordenadas de origen y destino
                is_transfer = (get_value(activity, 'category', '') == 'transfer' or 
                              get_value(activity, 'type', '') == 'transfer')
                
                base_data = {
                    "id": str(uuid.uuid4()),
                    "name": get_value(activity, 'name', 'Lugar sin nombre'),
                    "category": get_value(activity, 'place_type', get_value(activity, 'type', 'point_of_interest')),
                    "rating": get_value(activity, 'rating', 4.5) or 4.5,
                    "image": get_value(activity, 'image', ''),
                    "description": get_value(activity, 'description', f"Actividad en {get_value(activity, 'name', 'lugar')}"),
                    "estimated_time": f"{(calculate_dynamic_duration_with_context(activity, all_activities or [], activity_index or 0) if all_activities and activity_index is not None else calculate_dynamic_duration(activity))/60:.1f}h",
                    "priority": get_value(activity, 'priority', 5),
                    "lat": get_value(activity, 'lat', 0.0),
                    "lng": get_value(activity, 'lon', 0.0),  # Frontend espera 'lng'
                    "recommended_duration": f"{get_value(activity, 'duration_minutes', 60)/60:.1f}h",
                    "best_time": format_time_window(activity, all_activities, activity_index),
                    "order": order,
                    "is_intercity": False,
                    "quality_flag": get_value(activity, 'quality_flag', None),  # Agregar quality flag al frontend
                    "suggested": get_value(activity, 'suggested', False),  # Campo para lugares sugeridos automáticamente
                    "suggestion_reason": get_value(activity, 'suggestion_reason', None)  # Razón de la sugerencia
                }
                
                # Para transfers, agregar coordenadas de origen y destino
                if is_transfer:
                    # Obtener coordenadas FROM del optimizer (si están disponibles)
                    from_lat = get_value(activity, 'from_lat', 0.0)
                    from_lng = get_value(activity, 'from_lon', 0.0)  # Intenta 'from_lon' primero
                    if from_lng == 0.0:  # Si no encuentra, intenta 'from_lng'
                        from_lng = get_value(activity, 'from_lng', 0.0)
                    
                    # Si no hay coordenadas FROM del optimizer, calcular desde el place anterior
                    if (from_lat == 0.0 and from_lng == 0.0 and all_activities and activity_index is not None):
                        if activity_index > 0:
                            # Caso normal: tomar del place anterior en el mismo día
                            prev_activity = all_activities[activity_index - 1]
                            prev_lat = get_value(prev_activity, 'lat', 0.0)
                            # Intentar ambos formatos para longitude
                            prev_lng = get_value(prev_activity, 'lng', 0.0)
                            if prev_lng == 0.0:
                                prev_lng = get_value(prev_activity, 'lon', 0.0)
                            
                            # Debug: verificar si el lugar anterior tiene coordenadas válidas
                            if prev_lat == 0.0 and prev_lng == 0.0:
                                # Si el anterior tampoco tiene coordenadas, buscar más atrás
                                for j in range(activity_index - 2, -1, -1):
                                    candidate = all_activities[j]
                                    cand_lat = get_value(candidate, 'lat', 0.0)
                                    # Intentar ambos formatos para longitude
                                    cand_lng = get_value(candidate, 'lng', 0.0)
                                    if cand_lng == 0.0:
                                        cand_lng = get_value(candidate, 'lon', 0.0)
                                    if cand_lat != 0.0 or cand_lng != 0.0:
                                        prev_lat = cand_lat
                                        prev_lng = cand_lng
                                        break
                            
                            from_lat = prev_lat
                            from_lng = prev_lng
                        elif activity_index == 0 and hasattr(format_activity_for_frontend, '_day_data'):
                            # Caso especial: primer transfer del día, buscar en día anterior
                            day_data = format_activity_for_frontend._day_data
                            current_day = day_data.get('current_day', 1)
                            
                            if current_day > 1:
                                # Buscar el último place del día anterior
                                prev_day_activities = day_data.get('prev_day_activities', [])
                                if prev_day_activities:
                                    last_prev_activity = prev_day_activities[-1]
                                    # Si el último era un transfer, usar su destino (to_lat/to_lng)
                                    if get_value(last_prev_activity, 'category') == 'transfer':
                                        from_lat = get_value(last_prev_activity, 'lat', 0.0)  # Destino del transfer anterior
                                        from_lng = get_value(last_prev_activity, 'lng', 0.0)
                                        if from_lng == 0.0:
                                            from_lng = get_value(last_prev_activity, 'lon', 0.0)
                                    else:
                                        from_lat = get_value(last_prev_activity, 'lat', 0.0)
                                        from_lng = get_value(last_prev_activity, 'lng', 0.0)
                                        if from_lng == 0.0:
                                            from_lng = get_value(last_prev_activity, 'lon', 0.0)
                    
                    base_data.update({
                        "from_lat": from_lat,
                        "from_lng": from_lng,
                        "to_lat": get_value(activity, 'lat', 0.0),  # Destino
                        "to_lng": get_value(activity, 'lon', 0.0),   # Destino (frontend espera 'lng')
                        "from_place": get_value(activity, 'from_place', ''),
                        "to_place": get_value(activity, 'to_place', ''),
                        "distance_km": get_value(activity, 'distance_km', 0.0),
                        "transport_mode": get_value(activity, 'recommended_mode', 'walk')
                    })
                
                return base_data
        
        # 🎯 CREAR DÍAS VACÍOS CON SUGERENCIAS SI ES NECESARIO
        if suggestions_by_date:
            # Obtener fechas de días existentes
            existing_dates = {day.get("date") for day in days_data}
            
            # Crear días vacíos con sugerencias
            for suggestion_date, day_suggestions in suggestions_by_date.items():
                if suggestion_date not in existing_dates:
                    # Crear actividades sugeridas
                    suggested_activities = []
                    for idx, suggestion in enumerate(day_suggestions):
                        suggested_activities.append({
                            "name": suggestion["name"],
                            "lat": suggestion["lat"], 
                            "lon": suggestion["lon"],
                            "category": suggestion["category"],
                            "rating": suggestion["rating"],
                            "address": suggestion["address"],
                            "estimated_time": "1.5h",  # Tiempo estimado por defecto
                            "suggested": True,
                            "suggestion_reason": suggestion["suggestion_reason"]
                        })
                    
                    # Crear día vacío con sugerencias
                    empty_day_with_suggestions = {
                        "date": suggestion_date,
                        "activities": suggested_activities,
                        "free_minutes": max(0, 540 - len(suggested_activities) * 90)  # 9h - tiempo actividades
                    }
                    
                    days_data.append(empty_day_with_suggestions)
                    logger.info(f"📅 Día vacío creado para {suggestion_date} con {len(suggested_activities)} sugerencias")
            
            # Ordenar días por fecha
            days_data.sort(key=lambda x: x.get("date", ""))

        # Convertir días a formato frontend
        itinerary_days = []
        day_counter = 1
        prev_day_activities = []
        
        for day in days_data:
            # Configurar información de contexto para transfers intercity
            format_activity_for_frontend._day_data = {
                'current_day': day_counter,
                'prev_day_activities': prev_day_activities
            }
            
            # Separar places y transfers
            frontend_places = []
            day_transfers = []
            activities = day.get("activities", [])
            place_order = 1
            transfer_order = 1
            
            for idx, activity in enumerate(activities):
                # Detectar si es un transfer
                is_transfer = (get_value(activity, 'category', '') == 'transfer' or 
                              get_value(activity, 'type', '') == 'transfer' or
                              'traslado' in str(get_value(activity, 'name', '')).lower() or
                              'viaje' in str(get_value(activity, 'name', '')).lower() or
                              'regreso' in str(get_value(activity, 'name', '')).lower())
                
                if is_transfer:
                    # Agregar a transfers del día
                    transfer_data = format_activity_for_frontend(activity, transfer_order, activities, idx)
                    if transfer_data is not None:
                        # Cambiar el campo 'order' por 'transfer_order' para claridad
                        transfer_data['transfer_order'] = transfer_order
                        del transfer_data['order']  # Remover el campo 'order' original
                        day_transfers.append(transfer_data)
                        transfer_order += 1
                else:
                    # Agregar a places del día
                    place_data = format_activity_for_frontend(activity, place_order, activities, idx)
                    if place_data is not None:
                        frontend_places.append(place_data)
                        place_order += 1
            
            # Guardar actividades de este día para el siguiente (incluir ambos types)
            prev_day_activities = frontend_places + day_transfers
            
            # Calcular tiempos del día correctamente desde frontend_places
            total_activity_time_min = 0
            transport_time_min = 0
            walking_time_min = 0
            
            for place in frontend_places:
                estimated_hours = float(place.get('estimated_time', '0h').replace('h', ''))
                estimated_minutes = estimated_hours * 60
                
                # Sumar al tiempo total
                total_activity_time_min += estimated_minutes
                
                # Clasificar entre transporte y actividades
                is_transfer = (place.get('category') == 'transfer' or 
                              'traslado' in place.get('name', '').lower() or
                              'viaje' in place.get('name', '').lower() or
                              'regreso' in place.get('name', '').lower())
                
                if is_transfer:
                    # Clasificar entre walking y transport basado en duración
                    if estimated_minutes <= 30:  # <= 30min = walking
                        walking_time_min += estimated_minutes
                    else:  # > 30min = transport
                        transport_time_min += estimated_minutes
            
            # Formatear tiempos
            total_time_hours = total_activity_time_min / 60
            walking_time = f"{int(walking_time_min)}min" if walking_time_min < 60 else f"{int(walking_time_min//60)}h{int(walking_time_min%60)}min" if walking_time_min%60 > 0 else f"{int(walking_time_min//60)}h"
            transport_time = f"{int(transport_time_min)}min" if transport_time_min < 60 else f"{int(transport_time_min//60)}h{int(transport_time_min%60)}min" if transport_time_min%60 > 0 else f"{int(transport_time_min//60)}h"
            
            # Calcular tiempo libre (horas del día - tiempo total)
            daily_start_hour = request.daily_start_hour if hasattr(request, 'daily_start_hour') else 9
            daily_end_hour = request.daily_end_hour if hasattr(request, 'daily_end_hour') else 18
            available_hours = daily_end_hour - daily_start_hour
            free_hours = max(0, available_hours - total_time_hours)
            free_time = f"{int(free_hours)}h{int((free_hours % 1) * 60)}min" if free_hours % 1 > 0 else f"{int(free_hours)}h"
            
            # Determinar si es sugerido (días libres detectados)
            is_suggested = len(day.get("activities", [])) == 0
            
            day_data = {
                "day": day_counter,
                "date": day.get("date", ""),
                "places": frontend_places,
                "transfers": day_transfers,
                "total_places": len(frontend_places),
                "total_transfers": len(day_transfers),
                "total_time": f"{total_time_hours:.1f}h",
                "walking_time": walking_time,
                "transport_time": transport_time,  # Ahora separado correctamente
                "free_time": free_time,
                "is_suggested": is_suggested,
                "is_tentative": False
            }
            
            # Base si existe (campo opcional para V3.1)
            if day.get("base"):
                day_data["base"] = day["base"]
            if day.get("free_blocks"):
                day_data["free_blocks"] = day["free_blocks"]
            
            itinerary_days.append(day_data)
            day_counter += 1
        
        # 📊 RECALCULAR MÉTRICAS GLOBALES SUMANDO LOS DÍAS
        total_transport_minutes = 0
        total_walking_minutes = 0
        
        # Convertir strings como "24min" o "1h30min" a minutos
        def parse_time_string(time_str):
            if not time_str or time_str == "0min":
                return 0
            minutes = 0
            if "h" in time_str:
                parts = time_str.split("h")
                hours = int(parts[0]) if parts[0] else 0
                minutes += hours * 60
                if len(parts) > 1 and parts[1]:
                    min_part = parts[1].replace("min", "")
                    if min_part:
                        minutes += int(min_part)
            elif "min" in time_str:
                minutes = int(time_str.replace("min", ""))
            return minutes
        
        for day in itinerary_days:
            # Extraer minutos de transport_time y walking_time
            transport_str = day.get("transport_time", "0min")
            walking_str = day.get("walking_time", "0min")
            
            total_transport_minutes += parse_time_string(transport_str)
            total_walking_minutes += parse_time_string(walking_str)
        
        total_travel_minutes = total_transport_minutes + total_walking_minutes
        
        # Estructura final para frontend
        # 📊 MÉTRICAS COMPLETAS del optimizer (incluyendo optimization_mode, fallback_active, etc.)
        # NOTA: Los aliases ya fueron corregidos temprano, usar optimization_metrics directamente
        optimizer_metrics = optimization_metrics
        
        # � CORREGIR ALIASES EN INTERCITY TRANSFERS - usar nombres reales de bases
        if 'intercity_transfers' in optimizer_metrics and itinerary_days:
            corrected_transfers = []
            
            for transfer in optimizer_metrics['intercity_transfers']:
                corrected_transfer = transfer.copy()
                
                # Buscar la base real del día de origen usando coordenadas
                from_lat = transfer.get('from_lat', 0)
                from_lon = transfer.get('from_lon', 0)
                
                # Buscar la base real del día de destino usando coordenadas  
                to_lat = transfer.get('to_lat', 0)
                to_lon = transfer.get('to_lon', 0)
                
                # Corregir nombre FROM usando bases reales
                for day in itinerary_days:
                    base = day.get('base', {})
                    if base:
                        base_lat = base.get('lat', 0)
                        base_lon = base.get('lon', 0)
                        
                        # Si las coordenadas coinciden con FROM, usar el nombre real
                        if (abs(base_lat - from_lat) < 0.01 and 
                            abs(base_lon - from_lon) < 0.01):
                            corrected_transfer['from'] = base.get('name', transfer.get('from', ''))
                        
                        # Si las coordenadas coinciden con TO, usar el nombre real
                        if (abs(base_lat - to_lat) < 0.01 and 
                            abs(base_lon - to_lon) < 0.01):
                            corrected_transfer['to'] = base.get('name', transfer.get('to', ''))
                
                corrected_transfers.append(corrected_transfer)
            
            # Reemplazar los transfers corregidos
            optimizer_metrics['intercity_transfers'] = corrected_transfers
        
        #  Calcular duración del procesamiento
        duration = time_module.time() - start_time
        
        # 🔧 CORREGIR ALIASES EN day['transfers'] TAMBIÉN
        for day in itinerary_days:
            if 'transfers' in day:
                for transfer in day['transfers']:
                    if transfer.get('type') == 'intercity_transfer':
                        # Obtener coordenadas del transfer
                        from_lat = transfer.get('from_lat', 0)
                        from_lon = transfer.get('from_lon', 0)
                        to_lat = transfer.get('to_lat', 0) 
                        to_lon = transfer.get('to_lon', 0)
                        
                        # Corregir nombres usando bases reales
                        for check_day in itinerary_days:
                            base = check_day.get('base', {})
                            if base:
                                base_lat = base.get('lat', 0)
                                base_lon = base.get('lon', 0)
                                
                                # Corregir FROM
                                if (abs(base_lat - from_lat) < 0.01 and 
                                    abs(base_lon - from_lon) < 0.01):
                                    transfer['from'] = base.get('name', transfer.get('from', ''))
                                
                                # Corregir TO
                                if (abs(base_lat - to_lat) < 0.01 and 
                                    abs(base_lon - to_lon) < 0.01):
                                    transfer['to'] = base.get('name', transfer.get('to', ''))
        
        # 🧠 OBTENER INFORMACIÓN SEMÁNTICA GLOBAL
        semantic_info = get_semantic_status()
        
        formatted_result = {
            "itinerary": itinerary_days,
            "optimization_metrics": {
                # Métricas del optimizer (incluye optimization_mode, fallback_active, intercity_transfers, etc.)
                **optimizer_metrics,
                # Métricas adicionales calculadas en el API (recalculadas desde días)
                "total_distance_km": optimizer_metrics.get("total_distance_km", 0),
                "total_travel_time_minutes": int(total_travel_minutes),
                "transport_time_minutes": total_transport_minutes,  # Suma de transporte
                "walking_time_minutes": total_walking_minutes,     # Suma de caminata
                "processing_time_seconds": round(duration, 2),
                "hotels_provided": hotels_provided,
                "hotels_count": len(accommodations_data) if accommodations_data else 0,
                # Override el optimization_mode si se usaron hoteles
                "optimization_mode": "hotel_centroid" if hotels_provided else optimizer_metrics.get("optimization_mode", "geographic_v31"),
                # 🧠 INFORMACIÓN SEMÁNTICA
                "semantic_enabled": semantic_info['enabled'],
                "semantic_features_used": semantic_info['features'] if semantic_info['enabled'] else None,
                "analysis_type": "semantic_enhanced" if semantic_info['enabled'] else "geographic_basic"
            },
            "recommendations": base_recommendations
        }
        
        # 🧠 GENERAR RECOMENDACIONES AUTOMÁTICAS PARA DÍAS LIBRES
        auto_recommendations = []
        
        # ⚠️ GENERAR RECOMENDACIONES PARA LUGARES CON QUALITY FLAGS
        quality_recommendations = []
        for day in itinerary_days:
            for place in day.get("places", []):
                if place.get("quality_flag") == "user_provided_below_threshold":
                    # Lugar proporcionado por usuario con rating < 4.5
                    place_name = place.get("name", "lugar")
                    place_rating = place.get("rating", 0)
                    quality_recommendations.append(
                        f"⚠️ '{place_name}' ({place_rating}⭐) tiene rating bajo. "
                        f"Considera alternativas cercanas con mejor valoración."
                    )
        
        if quality_recommendations:
            base_recommendations.extend(quality_recommendations)
            
        # Combinar recomendaciones automáticas con las base
        if auto_recommendations:
            base_recommendations.extend(auto_recommendations)
        
        # 1. Detectar días completamente vacíos (sin actividades)
        empty_days = []
        total_days_requested = (request.end_date - request.start_date).days + 1
        days_with_activities = len(days_data)
        
        logger.info(f"🔍 DÍAS LIBRES DEBUG: total_solicitados={total_days_requested}, días_con_actividades={days_with_activities}")
        logger.info(f"📅 Fechas existentes en days_data: {[day.get('date', 'NO_DATE') for day in days_data]}")
        
        if days_with_activities < total_days_requested:
            # Generar fechas faltantes
            from datetime import timedelta
            current_date = request.start_date
            existing_dates = {day["date"] for day in days_data}
            
            for i in range(total_days_requested):
                date_str = current_date.strftime('%Y-%m-%d')
                if date_str not in existing_dates:
                    empty_days.append({
                        "date": date_str,
                        "free_minutes": 540,  # 9 horas completas (9:00-18:00)
                        "activities_count": 0,
                        "type": "completely_free"
                    })
                current_date += timedelta(days=1)
        
        # 2. Detectar días con poco contenido o tiempo libre excesivo
        partial_free_days = []
        for day in days_data:
            free_minutes = day.get("free_minutes", 0)
            activities_count = len(day.get("activities", []))
            
            # Criterios para día "libre" o con espacio para más actividades
            if free_minutes > 120 or activities_count <= 3:  # Más de 2h libres o pocas actividades
                partial_free_days.append({
                    "date": day["date"],
                    "free_minutes": free_minutes,
                    "activities_count": activities_count,
                    "existing_activities": day.get("activities", []),
                    "type": "partially_free"
                })
        
        # Combinar ambos tipos de días libres
        free_days_detected = empty_days + partial_free_days
        
        logger.info(f"🏖️ RESUMEN DÍAS LIBRES: empty_days={len(empty_days)}, partial_free_days={len(partial_free_days)}, total={len(free_days_detected)}")
        if empty_days:
            logger.info(f"📅 Días vacíos detectados: {[d['date'] for d in empty_days]}")
        if partial_free_days:
            logger.info(f"📅 Días parciales detectados: {[d['date'] for d in partial_free_days]}")
        
        # Inicializar diccionario de sugerencias
        suggestions_by_date = {}
        
        # 🎯 GENERAR SUGERENCIAS AUTOMÁTICAS PARA DÍAS LIBRES
        if free_days_detected:
            logger.info(f"🏖️ Detectados {len(free_days_detected)} días libres - generando sugerencias automáticas")
            
            try:
                from services.google_places_service import GooglePlacesService
                places_service = GooglePlacesService()
                
                # Obtener centro geográfico de los lugares existentes para contexto
                if request.places:
                    center_lat = sum(p.lat for p in request.places) / len(request.places)
                    center_lon = sum(p.lon for p in request.places) / len(request.places)
                else:
                    # Fallback: usar primer hotel si existe
                    if accommodations_data and accommodations_data[0]:
                        center_lat = accommodations_data[0].get('lat', 40.7128)
                        center_lon = accommodations_data[0].get('lon', -74.0060)
                    else:
                        # Default: NYC como fallback
                        center_lat, center_lon = 40.7128, -74.0060
                
                suggestions_generated = []
                
                for free_day in free_days_detected:
                    day_date = free_day["date"]
                    day_type = free_day["type"]
                    
                    logger.info(f"📍 Generando sugerencias para {day_date} ({day_type})")
                    
                    # Generar 3-5 sugerencias por día libre (sin restaurantes)
                    place_types = ["tourist_attraction", "museum", "park", "art_gallery", "shopping_mall", "point_of_interest"]
                    day_suggestions = []
                    
                    for place_type in place_types:
                        try:
                            suggestions = await places_service.search_nearby(
                                lat=center_lat,
                                lon=center_lon,
                                types=[place_type],
                                radius_m=5000,  # 5km radius
                                limit=2
                            )
                            
                            for suggestion in suggestions[:2]:  # Max 2 por tipo
                                # Filtrar restaurantes y hoteles explícitamente
                                suggestion_type = suggestion.get("type", "").lower()
                                suggestion_category = suggestion.get("category", "").lower()
                                
                                # Excluir si es restaurante, hotel o accommodation
                                if any(excluded in suggestion_type or excluded in suggestion_category 
                                       for excluded in ["restaurant", "cafe", "food", "hotel", "accommodation", "lodging"]):
                                    continue
                                
                                day_suggestions.append({
                                    "name": suggestion.get("name", "Lugar sugerido"),
                                    "rating": suggestion.get("rating", 4.0),
                                    "address": suggestion.get("address", "Dirección no disponible"),
                                    "category": place_type,
                                    "lat": suggestion.get("lat", center_lat),
                                    "lon": suggestion.get("lon", center_lon),
                                    "suggested": True,
                                    "suggestion_reason": f"Sugerido para día libre {day_date}"
                                })
                            
                        except Exception as e:
                            logger.warning(f"Error generando sugerencias tipo {place_type}: {e}")
                    
                    if day_suggestions:
                        suggestions_generated.append({
                            "date": day_date,
                            "suggestions": day_suggestions[:5],  # Max 5 por día
                            "type": day_type
                        })
                        
                        logger.info(f"✅ {len(day_suggestions[:5])} sugerencias generadas para {day_date}")
                
                # Almacenar sugerencias para incluir en días vacíos
                if suggestions_generated:
                    for day_suggestions in suggestions_generated:
                        suggestions_by_date[day_suggestions["date"]] = day_suggestions["suggestions"]
                    
                    suggestion_text = []
                    for day_suggestions in suggestions_generated:
                        day_date = day_suggestions["date"]
                        suggestions_count = len(day_suggestions["suggestions"])
                        suggestion_text.append(
                            f"📅 {day_date}: {suggestions_count} lugares sugeridos disponibles"
                        )
                    
                    auto_recommendations.extend([
                        "🎯 Días libres detectados - sugerencias automáticas generadas:",
                        *suggestion_text,
                        "💡 Las sugerencias aparecen marcadas como 'suggested: true'"
                    ])
                    
                    logger.info(f"🎉 Sugerencias automáticas completadas: {len(suggestions_generated)} días procesados")
                
            except Exception as e:
                logger.error(f"❌ Error generando sugerencias automáticas: {e}")
                auto_recommendations.append(
                    "⚠️ Error generando sugerencias automáticas para días libres. "
                    "Contacta soporte si persiste el problema."
                )
        
        # Log success
        # analytics.track_request(f"hybrid_itinerary_{optimization_mode}_success", {
        #     "efficiency_score": optimization_result.get("optimization_metrics", {}).get("efficiency_score", 0.9),
        #     "total_activities": total_activities,
        #     "days_used": len(days_data),
        #     "processing_time_seconds": round(duration, 2),
        #     "optimization_mode": optimization_mode,
        #     "hotels_provided": hotels_provided,
        #     "hotels_count": len(accommodations_data) if accommodations_data else 0
        # })
        
        if hotels_provided:
            logging.info(f"✅ Optimización híbrida CON HOTELES completada en {duration:.2f}s")
            logging.info(f"🏨 {len(accommodations_data)} hoteles usados como centroides")
        else:
            logging.info(f"✅ Optimización híbrida GEOGRÁFICA completada en {duration:.2f}s")
            
        logging.info(f"🎯 Resultado: {total_activities} actividades, score {optimization_result.get('optimization_metrics', {}).get('efficiency_score', 0.9):.1%}")
        
        return ItineraryResponse(**formatted_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ Error generating hybrid itinerary: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error generating hybrid itinerary: {str(e)}"
        )

# ===== MULTI-CIUDAD ENDPOINTS =====

@app.post("/api/v3/multi-city/analyze", response_model=MultiCityAnalysisResponse, tags=["Multi-Ciudad"])
async def analyze_multi_city_feasibility(request: MultiCityAnalysisRequest):
    """
    🌍 Análisis de viabilidad multi-ciudad
    
    Analiza un conjunto de POIs para determinar:
    - Número de ciudades/países detectados
    - Complejidad del viaje 
    - Duración recomendada
    - Estrategia de optimización sugerida
    """
    try:
        start_time = time_module.time()
        
        # Convertir places a formato interno
        pois = []
        for place in request.places:
            poi_data = {
                'name': place.name,
                'lat': place.lat,
                'lon': place.lon,
                'category': 'attraction'
            }
            
            # Añadir campos opcionales si existen
            if hasattr(place, 'city') and place.city:
                poi_data['city'] = place.city
            if hasattr(place, 'country') and place.country:
                poi_data['country'] = place.country
            if hasattr(place, 'category') and place.category:
                poi_data['category'] = place.category
                
            pois.append(poi_data)
        
        # Inicializar servicios
        clustering_service = CityClusteringService()
        
        # Clustering de ciudades
        city_clusters = clustering_service.cluster_pois_advanced(pois)
        
        # Análisis de complejidad usando InterCity Service
        from services.intercity_service import InterCityService
        intercity_service = InterCityService()
        
        # Convertir clusters a Cities
        cities = []
        for cluster in city_clusters:
            from services.intercity_service import City
            city = City(
                name=cluster.name,
                center_lat=cluster.center_lat,
                center_lon=cluster.center_lon,
                country=cluster.country,
                pois=cluster.pois
            )
            cities.append(city)
        
        # Análisis de complejidad
        analysis = intercity_service.analyze_multi_city_complexity(cities)
        
        processing_time = (time_module.time() - start_time) * 1000
        
        # Calcular score de viabilidad
        feasibility_score = min(1.0, 1.0 - (analysis.get('max_intercity_distance_km', 0) / 3000.0))
        
        # Generar warnings
        warnings = []
        if analysis.get('max_intercity_distance_km', 0) > 1500:
            warnings.append("Distancias muy largas entre ciudades - considerar vuelos")
        if analysis.get('total_countries', 0) > 3:
            warnings.append("Viaje multi-país complejo - requiere planificación avanzada")
        if len(pois) > 20:
            warnings.append("Muchos POIs - considerar extender duración del viaje")
            
        # Mapear complexity a valores válidos
        complexity_map = {
            'simple_intercity': 'simple',
            'medium_intercity': 'intercity', 
            'complex_intercity': 'international',
            'international_complex': 'international_complex'
        }
        
        return MultiCityAnalysisResponse(
            cities_detected=analysis['total_cities'],
            countries_detected=analysis.get('total_countries', 1),
            max_intercity_distance_km=analysis.get('max_intercity_distance_km', 0),
            complexity_level=complexity_map.get(analysis['complexity'], 'complex'),
            recommended_duration_days=analysis.get('estimated_trip_days', 7),
            optimization_recommendation=analysis['recommendation'],
            feasibility_score=feasibility_score,
            warnings=warnings
        )
        
    except Exception as e:
        logger.error(f"Error en análisis multi-ciudad: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing multi-city feasibility: {str(e)}"
        )

@app.post("/api/v3/multi-city/optimize", response_model=MultiCityItineraryResponse, tags=["Multi-Ciudad"])
async def optimize_multi_city_itinerary(request: MultiCityOptimizationRequest):
    """
    🎯 Optimización completa de itinerario multi-ciudad
    
    Genera un itinerario optimizado usando la arquitectura grafo-de-grafos:
    - Clustering automático de POIs por ciudades
    - Optimización de secuencia intercity (TSP)
    - Distribución inteligente de días por ciudad
    - Planificación de accommodations
    - Análisis logístico completo
    """
    try:
        start_time = time_module.time()
        
        # Convertir request a formato interno
        pois = []
        for place in request.places:
            poi_data = {
                'name': place.name,
                'lat': place.lat,
                'lon': place.lon,
                'category': 'attraction',
                'visit_duration_hours': 2
            }
            
            # Añadir campos opcionales si existen
            if hasattr(place, 'city') and place.city:
                poi_data['city'] = place.city
            if hasattr(place, 'country') and place.country:
                poi_data['country'] = place.country
            if hasattr(place, 'category') and place.category:
                poi_data['category'] = place.category
            if hasattr(place, 'visit_duration_hours'):
                poi_data['visit_duration_hours'] = place.visit_duration_hours
                
            pois.append(poi_data)
        
        # Inicializar optimizador multi-ciudad
        optimizer = MultiCityOptimizerSimple()
        
        # Optimización principal
        itinerary = optimizer.optimize_multi_city_itinerary(
            pois=pois,
            trip_duration_days=request.duration_days,
            start_city=request.start_city
        )
        
        # Planificación de accommodations si se solicita
        accommodations_info = []
        estimated_cost = 0.0
        
        if request.include_accommodations:
            hotel_service = HotelRecommender()
            
            # Preparar ciudades para hotel service
            cities_for_hotels = []
            for city in itinerary.cities:
                cities_for_hotels.append({
                    'name': city.name,
                    'pois': city.pois,
                    'coordinates': city.coordinates
                })
            
            # Calcular días por ciudad
            days_per_city = {}
            for city in itinerary.cities:
                city_days = sum(
                    1 for day_pois in itinerary.daily_schedules.values()
                    if any(poi.get('city') == city.name for poi in day_pois)
                )
                days_per_city[city.name] = max(1, city_days)
            
            # Planificar accommodations
            accommodation_plan = hotel_service.plan_multi_city_accommodations(
                cities_for_hotels, days_per_city
            )
            
            # Convertir a formato de respuesta
            for acc in accommodation_plan.accommodations:
                hotel = acc['hotel']
                accommodations_info.append({
                    'city': acc['city'],
                    'hotel_name': hotel.name,
                    'rating': hotel.rating,
                    'price_range': hotel.price_range,
                    'nights': acc['nights'],
                    'check_in_day': acc['check_in_day'],
                    'check_out_day': acc['check_out_day'],
                    'estimated_cost_usd': 120.0 * acc['nights'],  # Estimación básica
                    'coordinates': {
                        'latitude': hotel.lat,
                        'longitude': hotel.lon
                    }
                })
            
            estimated_cost = accommodation_plan.estimated_cost
        
        # Convertir ciudades a formato de respuesta
        cities_info = []
        for city in itinerary.cities:
            cities_info.append({
                'name': city.name,
                'country': city.country,
                'coordinates': {
                    'latitude': city.center_lat,
                    'longitude': city.center_lon
                },
                'pois_count': len(city.pois),
                'assigned_days': sum(
                    1 for day_pois in itinerary.daily_schedules.values()
                    if any(poi.get('city') == city.name for poi in day_pois)
                )
            })
        
        processing_time = (time_module.time() - start_time) * 1000
        
        return MultiCityItineraryResponse(
            success=True,
            cities=cities_info,
            city_sequence=itinerary.get_city_sequence(),
            daily_schedule=itinerary.daily_schedules,
            accommodations=accommodations_info,
            total_duration_days=itinerary.total_duration_days,
            countries_count=itinerary.countries_count,
            total_distance_km=itinerary.total_distance_km,
            estimated_accommodation_cost_usd=estimated_cost,
            optimization_strategy=itinerary.optimization_strategy.value,
            confidence=itinerary.confidence,
            processing_time_ms=processing_time,
            logistics={
                'complexity': 'multi_city',
                'intercity_routes_count': len(itinerary.intercity_routes),
                'avg_pois_per_city': len(pois) / len(itinerary.cities) if itinerary.cities else 0
            }
        )
        
    except Exception as e:
        logger.error(f"Error en optimización multi-ciudad: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error optimizing multi-city itinerary: {str(e)}"
        )

@app.post("/recommend_hotels")
@app.post("/api/v2/hotels/recommend")
async def recommend_hotels_endpoint(request: HotelRecommendationRequest):
    """
    🏨 Recomendar hoteles basado en lugares a visitar
    
    Analiza la ubicación de los lugares del itinerario y recomienda
    hoteles óptimos basado en proximidad geográfica y conveniencia.
    """
    try:
        start_time = time_module.time()
        
        # Convertir lugares del request a formato interno
        places_data = []
        for place in request.places:
            places_data.append({
                'name': place.name,
                'lat': place.lat,
                'lon': place.lon,
                'type': place.type.value if hasattr(place.type, 'value') else str(place.type),
                'priority': getattr(place, 'priority', 5)
            })
        
        # Inicializar recomendador
        hotel_recommender = HotelRecommender()
        
        # Generar recomendaciones
        recommendations = await hotel_recommender.recommend_hotels(
            places_data,
            max_recommendations=request.max_recommendations,
            price_preference=request.price_preference
        )
        
        # Formatear recomendaciones como una lista
        formatted_hotels = []
        if recommendations:
            centroid = hotel_recommender.calculate_geographic_centroid(places_data)
            for hotel in recommendations:
                formatted_hotel = {
                    "name": hotel.name,
                    "lat": hotel.lat,
                    "lon": hotel.lon,
                    "address": hotel.address,
                    "rating": hotel.rating,
                    "price_range": hotel.price_range,
                    "convenience_score": hotel.convenience_score,
                    "type": "hotel",
                    "distance_to_centroid_km": hotel.distance_to_centroid_km,
                    "avg_distance_to_places_km": hotel.avg_distance_to_places_km,
                    "analysis": {
                        "places_analyzed": len(places_data),
                        "activity_centroid": {
                            "latitude": round(centroid[0], 6),
                            "longitude": round(centroid[1], 6)
                        }
                    }
                }
                formatted_hotels.append(formatted_hotel)
        
        # Métricas de rendimiento
        duration = time_module.time() - start_time
        
        # Añadir métricas de rendimiento a cada hotel
        for hotel in formatted_hotels:
            hotel["performance"] = {
                "processing_time_s": round(duration, 2),
                "generated_at": datetime.now().isoformat()
            }
        
        logging.info(f"🏨 Recomendaciones de hoteles generadas en {duration:.2f}s")
        logging.info(f"📊 Mejor opción: {recommendations[0].name if recommendations else 'Ninguna'}")
        
        return formatted_hotels  # Retornamos la lista de hoteles
        
    except Exception as e:
        logging.error(f"❌ Error recomendando hoteles: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error generating hotel recommendations: {str(e)}"
        )


# Monitoring endpoints moved to routers/monitoring.py


# Routing endpoints (drive/walk/bike/compare, cache, health/multimodal)
# moved to routers/routing.py

# Función auxiliar para calcular duración de visita
def calculate_visit_duration(place_type: str) -> int:
    """Calcular duración de visita por tipo de lugar (en minutos)"""
    duration_map = {
        'restaurant': 90,
        'museum': 120,
        'tourist_attraction': 90,
        'park': 60,
        'cafe': 45,
        'shopping_mall': 120,
        'hotel': 30,
        'church': 45,
        'art_gallery': 90,
        'zoo': 180,
        'aquarium': 150,
        'amusement_park': 240,
        'bar': 90,
        'night_club': 180,
        'library': 60,
        'movie_theater': 150
    }
    return duration_map.get(place_type.lower(), 60)  # 60 minutos por defecto

# Función auxiliar para formatear actividades para el frontend
def format_activity_for_frontend_simple(activity, order, activities=None, idx=None):
    """Versión simplificada para el endpoint multimodal"""
    import uuid
    
    def get_value(obj, key, default=None):
        """Función helper para extraer valores robusta"""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        else:
            # Es un objeto - usar getattr con manejo de errores
            try:
                return getattr(obj, key, default)
            except (AttributeError, TypeError):
                return default
    
    # Detectar si es un transfer intercity
    is_intercity = (get_value(activity, 'type') == 'intercity_transfer' or 
                   get_value(activity, 'activity_type') == 'intercity_transfer')
    
    if is_intercity:
        # Es un transfer - formato diferente
        start_time = get_value(activity, 'start_time')
        end_time = get_value(activity, 'end_time')
        
        transfer_data = {
            "id": str(uuid.uuid4()),
            "name": get_value(activity, 'name', 'Transfer'),
            "transfer_type": "intercity",
            "from_place": get_value(activity, 'from_place', 'Origen'),
            "to_place": get_value(activity, 'to_place', 'Destino'),
            "distance_km": get_value(activity, 'distance_km', 0.0),
            "duration_minutes": get_value(activity, 'duration_minutes', 0),
            "transport_mode": get_value(activity, 'recommended_mode', 'drive'),
            "order": order
        }
        
        # Agregar tiempos si están disponibles
        if start_time is not None:
            hours = start_time // 60
            minutes = start_time % 60
            transfer_data["departure_time"] = f"{hours:02d}:{minutes:02d}"
            
        if end_time is not None:
            hours = end_time // 60
            minutes = end_time % 60
            transfer_data["arrival_time"] = f"{hours:02d}:{minutes:02d}"
            
        return transfer_data
    else:
        # Es un lugar normal
        duration_min = get_value(activity, 'duration_minutes', 60)
        start_time = get_value(activity, 'start_time')
        end_time = get_value(activity, 'end_time')
        
        # Construir respuesta base
        place_data = {
            "id": str(uuid.uuid4()),
            "name": get_value(activity, 'name', 'Lugar sin nombre'),
            "category": get_value(activity, 'type', get_value(activity, 'place_type', 'point_of_interest')),
            "rating": get_value(activity, 'rating', 4.5) or 4.5,
            "image": get_value(activity, 'image', ''),
            "description": get_value(activity, 'description', f"Visita a {get_value(activity, 'name', 'lugar')}"),
            "estimated_time": f"{duration_min/60:.1f}h",
            "duration_minutes": duration_min,
            "priority": get_value(activity, 'priority', 5),
            "lat": get_value(activity, 'lat', 0.0),
            "lng": get_value(activity, 'lon', get_value(activity, 'lng', 0.0)),
            "recommended_duration": f"{duration_min/60:.1f}h",
            "order": order,
            "walking_time_minutes": 0  # Por ahora simplificado
        }
        
        # Agregar tiempos si están disponibles (formato HH:MM)
        if start_time is not None:
            hours = start_time // 60
            minutes = start_time % 60
            place_data["arrival_time"] = f"{hours:02d}:{minutes:02d}"
            
        if end_time is not None:
            hours = end_time // 60
            minutes = end_time % 60
            place_data["departure_time"] = f"{hours:02d}:{minutes:02d}"
            
        return place_data

@app.post("/itinerary/multimodal", response_model=ItineraryResponse, tags=["Multi-Modal Itinerary"])
async def generate_multimodal_itinerary_endpoint(request: ItineraryRequest):
    """
    🎯 Generar itinerario completo usando sistema multi-modal mejorado
    
    Este endpoint combina:
    - HybridOptimizerV31 para la planificación inteligente
    - ChileMultiModalRouter para rutas precisas
    - Optimización de transporte según distancias y condiciones
    """
    try:
        start_time = time_module.time()
        logger.info(f"🚀 Iniciando generación de itinerario multi-modal")
        logger.info(f"📍 {len(request.places)} lugares, modo: {request.transport_mode}")
        
        # Convertir fechas
        start_date = request.start_date if isinstance(request.start_date, datetime) else datetime.strptime(str(request.start_date), '%Y-%m-%d')
        end_date = request.end_date if isinstance(request.end_date, datetime) else datetime.strptime(str(request.end_date), '%Y-%m-%d')
        
        # Validaciones básicas
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="La fecha de inicio debe ser anterior a la fecha de fin")
        
        # Normalizar lugares de entrada — separar alojamientos de actividades
        LODGING_TYPES = {'hotel', 'lodging', 'accommodation', 'motel', 'hostel', 'resort'}
        # Pre-fetch learned dwell stats once per request (see dwell_stats_service).
        raw_place_dicts = [
            (place.dict() if hasattr(place, 'dict') else place)
            for place in request.places
        ]
        dwell_lookup = await build_dwell_stats_lookup(raw_place_dicts)

        normalized_places = []
        for place_dict in raw_place_dicts:
            place_type = (place_dict.get('type') or place_dict.get('category') or '').lower()

            # Hotels go as accommodation, never as activity POIs
            if place_type in LODGING_TYPES:
                place_dict['type'] = 'accommodation'
                place_dict['place_type'] = 'accommodation'
                logger.info(f"🏨 Marcado como alojamiento (no actividad): {place_dict.get('name', 'N/A')}")

            # Duration fallback pyramid: user input → learned dwell stats →
            # hardcoded duration_map. Same logic as the v2 endpoint above.
            existing = place_dict.get('duration_minutes')
            if existing is None and place_dict.get('min_duration_hours') is not None:
                existing = int(round(float(place_dict['min_duration_hours']) * 60))
            if existing is not None:
                place_dict['duration_minutes'] = max(1, int(existing))
            else:
                learned = dwell_lookup.lookup(
                    place_id=(place_dict.get('id')
                              or place_dict.get('google_place_id')
                              or place_dict.get('place_id')),
                    category=(place_dict.get('category') or place_dict.get('type')),
                    lat=place_dict.get('lat'),
                    lon=place_dict.get('lon'),
                )
                place_dict['duration_minutes'] = learned if learned is not None \
                    else calculate_visit_duration(place_dict.get('type', 'point_of_interest'))
            normalized_places.append(place_dict)

        # Normalize accommodations to plain dicts. Downstream optimizers
        # (hybrid_optimizer_v31, hotel_recommender, ortools_service) expect
        # subscript access (`acc['lat']`, `acc['city']`, etc.). Pydantic v2
        # `Accommodation` instances are NOT subscriptable, so leaving them as
        # model objects raised `'Accommodation' object is not subscriptable`
        # and crashed the multi-modal endpoint with a 500.
        normalized_accommodations = []
        for acc in (request.accommodations or []):
            if hasattr(acc, 'model_dump'):
                normalized_accommodations.append(acc.model_dump())
            elif hasattr(acc, 'dict'):
                normalized_accommodations.append(acc.dict())
            else:
                normalized_accommodations.append(acc)

        # 🆕 Crear mapa de horarios personalizados por fecha
        custom_schedule_map = {}
        if request.custom_schedules:
            for schedule in request.custom_schedules:
                custom_schedule_map[schedule.date] = {
                    'start_hour': schedule.start_hour,
                    'end_hour': schedule.end_hour
                }
            logger.info(f"⏰ Horarios personalizados configurados para {len(custom_schedule_map)} días: {list(custom_schedule_map.keys())}")
            
        # Configurar extra_info para el optimizador
        extra_info = {
            'use_multimodal_router': True,
            'max_walking_distance_km': request.max_walking_distance_km,
            'max_daily_activities': request.max_daily_activities,
            'preferences': request.preferences or {},
            'multimodal_router_instance': get_chile_router(),
            'custom_schedules': custom_schedule_map  # 🆕 Agregar horarios personalizados
        }
        
        logger.info(f"🔧 Configuración multi-modal activada")
        
        # 🎯 DECISIÓN INTELIGENTE: ¿Usar ORTools o sistema legacy?
        decision = await should_use_city2graph(request)
        use_ortools = decision.get('use_city2graph', False)
        complexity_score = decision.get('complexity_score', 0.0)

        logger.info(f"🧠 Decisión algoritmo: {'ORTools' if use_ortools else 'Legacy'} (score: {complexity_score})")

        # 🏨 (Fase 2.5) Override: si el trip trae stays con check_in/check_out,
        # forzamos el path legacy híbrido. Razón:
        #   1. El path OR-Tools actual no incorpora anchoring por fecha
        #      (su optimizer es single-day TSP/VRP).
        #   2. El path legacy híbrido SÍ aplica `reanchor_clusters_by_dates`
        #      cuando hay stays fechados (Fase 2).
        # El override es no-op para clientes legacy sin fechas en stays,
        # así que no degrada ningún comportamiento previo.
        # Operate on the normalized dict list — `getattr` would silently
        # return None on a dict and the override below would never trigger.
        has_dated_stays = any(
            a.get('check_in') is not None and a.get('check_out') is not None
            for a in normalized_accommodations
        )
        if has_dated_stays and use_ortools:
            logger.info(
                "🏨 Override Fase 2.5: stays con fechas detectadas → "
                "forzando path legacy híbrido para aprovechar date-anchoring "
                f"(complexity_score={complexity_score} ignorado)"
            )
            use_ortools = False

        if use_ortools and settings.ENABLE_ORTOOLS:
            # 🚀 USAR ORTools para casos complejos (multi-ciudad, rutas largas)
            logger.info(f"🚀 Usando ORTools para optimización avanzada")
            
            try:
                from services.city2graph_ortools_service import get_ortools_service
                ortools_service = await get_ortools_service()
                
                # Preparar request para ORTools
                ortools_request = {
                    'places': normalized_places,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'daily_start_hour': request.daily_start_hour,
                    'daily_end_hour': request.daily_end_hour,
                    'transport_mode': str(request.transport_mode),
                    'accommodations': normalized_accommodations,
                    'preferences': request.preferences or {},
                    'max_walking_distance_km': request.max_walking_distance_km,
                    'max_daily_activities': request.max_daily_activities
                }
                
                optimization_result = await ortools_service.optimize_with_ortools(ortools_request)
                
                # Marcar que se usó ORTools
                if optimization_result:
                    optimization_result['optimization_mode'] = 'ortools_advanced'
                    optimization_result['decision_factors'] = decision.get('factors', {})
                    
            except Exception as e:
                logger.warning(f"⚠️ ORTools falló, usando fallback: {e}")
                use_ortools = False
        
        if not use_ortools:
            # 🔄 USAR sistema legacy para casos simples o fallback
            logger.info(f"🔄 Usando sistema legacy híbrido")
            
            from utils.hybrid_optimizer_v31 import optimize_itinerary_hybrid_v31
            
            optimization_result = await optimize_itinerary_hybrid_v31(
            normalized_places,
            start_date,
            end_date,
            request.daily_start_hour,
            request.daily_end_hour,
            request.transport_mode,
            normalized_accommodations,
            "balanced",  # packing_strategy
            extra_info
        )
        
        if not optimization_result or 'days' not in optimization_result:
            raise ValueError("Resultado de optimización multi-modal inválido")
            
        # Normalizar formato de 'days' devuelto por el optimizer.
        # Algunos optimizadores retornan un dict {date: day_obj}, otros una lista.
        raw_days = optimization_result['days']
        if isinstance(raw_days, dict):
            # Convertir dict->lista preservando orden de fechas
            try:
                # Ordenar por fecha (clave) para consistencia
                days_data = [raw_days[k] for k in sorted(raw_days.keys())]
            except Exception:
                days_data = list(raw_days.values())
        else:
            days_data = raw_days
        
        # Procesar días para frontend (usar la lógica existente)
        itinerary_days = []
        day_counter = 1
        prev_day_activities = []
        
        for day in days_data:
            # Separar places y transfers
            frontend_places = []
            day_transfers = []
            activities = day.get("activities", [])
            place_order = 1
            transfer_order = 1
            
            # Función auxiliar para extraer valores de manera robusta
            def get_activity_value(obj, key, default=None):
                """Función helper para extraer valores de actividades"""
                if obj is None:
                    return default
                if isinstance(obj, dict):
                    return obj.get(key, default)
                else:
                    try:
                        return getattr(obj, key, default)
                    except (AttributeError, TypeError):
                        return default
            
            for idx, activity in enumerate(activities):
                # Skip accommodation/hotel entries — they are base, not activities
                act_type = (get_activity_value(activity, 'type') or get_activity_value(activity, 'category') or '').lower()
                if act_type in {'hotel', 'lodging', 'accommodation', 'motel', 'hostel', 'resort'}:
                    continue

                # Detectar si es un transfer
                if (act_type == 'intercity_transfer' or
                    get_activity_value(activity, 'activity_type') == 'intercity_transfer'):
                    # Es un transfer
                    transfer_data = format_activity_for_frontend_simple(activity, transfer_order, activities, idx)
                    if transfer_data is not None:
                        transfer_data['transfer_order'] = transfer_order
                        if 'order' in transfer_data:
                            del transfer_data['order']  # Remover el campo 'order' original
                        day_transfers.append(transfer_data)
                        transfer_order += 1
                else:
                    # Agregar a places del día
                    place_data = format_activity_for_frontend_simple(activity, place_order, activities, idx)
                    if place_data is not None:
                        frontend_places.append(place_data)
                        place_order += 1
            
            # Guardar actividades de este día para el siguiente
            prev_day_activities = frontend_places + day_transfers
            
            # Calcular tiempos del día
            total_activity_time_min = sum(p.get('duration_minutes', 0) for p in frontend_places)
            transport_time_min = sum(t.get('duration_minutes', 0) for t in day_transfers)
            walking_time_min = sum(p.get('walking_time_minutes', 0) for p in frontend_places)
            
            total_time_hours = (total_activity_time_min + transport_time_min) / 60.0
            free_hours = ((request.daily_end_hour - request.daily_start_hour) * 60 - 
                         total_activity_time_min - transport_time_min) / 60.0
            free_time = f"{int(free_hours)}h{int((free_hours % 1) * 60)}min" if free_hours % 1 > 0 else f"{int(free_hours)}h"
            
            # Determinar si es sugerido (días libres detectados)
            is_suggested = len(day.get("activities", [])) == 0
            
            day_data = {
                "day": day_counter,
                "date": day.get("date", ""),
                "places": frontend_places,
                "transfers": day_transfers,
                "total_places": len(frontend_places),
                "total_transfers": len(day_transfers),
                "total_time": f"{total_time_hours:.1f}h",
                "free_time": free_time,
                "transport_time": f"{transport_time_min}min",
                "walking_time": f"{walking_time_min}min",
                "is_suggested": is_suggested,
                "base": day.get("base"),
                "free_blocks": day.get("free_blocks", []),
                "actionable_recommendations": day.get("actionable_recommendations", []),
                "schedule_info": day.get("schedule_info")  # 🆕 Preservar horarios personalizados
            }
            
            itinerary_days.append(day_data)
            day_counter += 1
        
        # Calcular métricas finales
        optimization_metrics = optimization_result.get('optimization_metrics', {})
        total_activities = sum(len(day['places']) for day in itinerary_days)
        total_transfers = sum(len(day['transfers']) for day in itinerary_days)
        
        # Calcular tiempos totales
        total_transport_minutes = 0
        total_walking_minutes = 0
        
        def parse_time_string(time_str):
            if not time_str or time_str == "0min":
                return 0
            if "h" in time_str and "min" in time_str:
                parts = time_str.replace("h", " ").replace("min", "").split()
                return int(parts[0]) * 60 + int(parts[1])
            elif "h" in time_str:
                return int(time_str.replace("h", "")) * 60
            elif "min" in time_str:
                return int(time_str.replace("min", ""))
            return 0
            
        for day in itinerary_days:
            transport_str = day.get("transport_time", "0min")
            walking_str = day.get("walking_time", "0min")
            
            total_transport_minutes += parse_time_string(transport_str)
            total_walking_minutes += parse_time_string(walking_str)
        
        total_travel_minutes = total_transport_minutes + total_walking_minutes
        
        duration = time_module.time() - start_time
        
        # Información sobre el router multi-modal usado
        router = get_chile_router()
        router_stats = router.get_performance_stats() if router else {}
        
        logger.info(f"✅ Itinerario multi-modal generado en {duration:.2f}s")
        logger.info(f"📊 {total_activities} lugares, {total_transfers} transfers")
        logger.info(f"🚀 Router stats: {router_stats}")
        
        logger.info("🎯 INICIANDO DETECCIÓN DE DÍAS LIBRES...")
        
        # 🎯 DETECTAR Y GENERAR SUGERENCIAS PARA DÍAS LIBRES
        base_recommendations = [
            f"Itinerario optimizado con sistema multi-modal",
            f"{total_activities} actividades distribuidas en {len(itinerary_days)} días",
            f"Tiempo total de viaje: {total_travel_minutes} minutos",
            f"Router multi-modal: {router_stats.get('cached_graphs', 0)} grafos en caché"
        ]
        
        # Detectar días completamente vacíos (sin actividades)
        empty_days = []
        total_days_requested = (end_date - start_date).days + 1
        days_with_activities = len(itinerary_days)
        
        # 🔄 Cache global para evitar sugerencias repetidas entre días
        used_place_ids_global = set()
        day_counter = 1
        
        logger.info(f"🔍 DÍAS LIBRES DEBUG: total_solicitados={total_days_requested}, días_con_actividades={days_with_activities}")
        logger.info(f"📅 Fechas existentes en itinerary: {[day.get('date', 'NO_DATE') for day in itinerary_days]}")
        
        if days_with_activities < total_days_requested:
            # Generar fechas faltantes
            from datetime import timedelta
            current_date = start_date
            existing_dates = {day.get("date") for day in itinerary_days}
            
            for i in range(total_days_requested):
                date_str = current_date.strftime('%Y-%m-%d')
                if date_str not in existing_dates:
                    empty_days.append({
                        "date": date_str,
                        "free_minutes": 540,  # 9 horas completas (9:00-18:00)
                        "activities_count": 0,
                        "type": "completely_free"
                    })
                current_date += timedelta(days=1)
        
        # Detectar días con poco contenido o tiempo libre excesivo
        partial_free_days = []
        for day in itinerary_days:
            free_time_str = day.get("free_time", "0h")
            try:
                free_hours = float(free_time_str.replace('h', ''))
                free_minutes = free_hours * 60
            except:
                free_minutes = 0
                
            activities_count = len(day.get("places", []))
            
            # Criterios para día "libre" o con espacio para más actividades
            if free_minutes > 120 or activities_count <= 1:  # Más de 2h libres o pocas actividades
                partial_free_days.append({
                    "date": day.get("date"),
                    "free_minutes": free_minutes,
                    "activities_count": activities_count,
                    "existing_activities": day.get("places", []),
                    "type": "partially_free"
                })
        
        # Combinar ambos tipos de días libres
        free_days_detected = empty_days + partial_free_days
        
        logger.info(f"🏖️ RESUMEN DÍAS LIBRES: empty_days={len(empty_days)}, partial_free_days={len(partial_free_days)}, total={len(free_days_detected)}")
        if empty_days:
            logger.info(f"📅 Días vacíos detectados: {[d['date'] for d in empty_days]}")
        if partial_free_days:
            logger.info(f"📅 Días parciales detectados: {[d['date'] for d in partial_free_days]}")
        
        # Generar sugerencias automáticas para días libres
        if free_days_detected:
            logger.info(f"🏖️ Detectados {len(free_days_detected)} días libres - generando sugerencias automáticas")
            
            try:
                from services.google_places_service import GooglePlacesService
                places_service = GooglePlacesService()
                
                # Obtener centro geográfico de los lugares existentes para contexto
                if request.places:
                    center_lat = sum(p.lat for p in request.places) / len(request.places)
                    center_lon = sum(p.lon for p in request.places) / len(request.places)
                    logger.info(f"📍 Centro calculado para sugerencias: {center_lat:.4f}, {center_lon:.4f} (basado en {len(request.places)} lugares)")
                else:
                    # Default: Orlando como fallback
                    center_lat, center_lon = 28.5383, -81.3792
                    logger.warning(f"📍 Usando coordenadas por defecto: {center_lat:.4f}, {center_lon:.4f}")
                
                # Detectar días existentes en el itinerario que están vacíos (0 places)
                existing_empty_days = [d for d in itinerary_days if len(d.get('places', [])) == 0]

                # Agregar/actualizar días vacíos con sugerencias al itinerario
                # 1) Reemplazar días existentes que están vacíos
                for existing in existing_empty_days:
                    day_date = existing.get("date")
                    logger.info(f"📍 Rellenando día existente vacío {day_date} con sugerencias")
                    # 🎯 Generar sugerencias REALES con Google Places para el día existente
                    suggested_places = await generate_smart_suggestions_for_day(
                        places_service, center_lat, center_lon, day_date, request, day_counter, used_place_ids_global
                    )
                    day_counter += 1
                    # Reemplazar el contenido del día existente
                    existing['places'] = suggested_places
                    existing['total_places'] = len(suggested_places)
                    existing['total_time'] = "3.5h"
                    existing['free_time'] = "5.5h"
                    existing['suggested_day'] = True
                    existing['suggestion_reason'] = f"Día libre detectado - sugerencias automáticas generadas"
                    logger.info(f"✅ Día existente {day_date} actualizado con {len(suggested_places)} sugerencias")

                # 2) Crear días completamente faltantes (fechas que no estaban en itinerary_days)
                for empty_day in empty_days:
                    day_date = empty_day["date"]
                    logger.info(f"📍 Creando día vacío con sugerencias para {day_date}")
                    
                    # 🎯 Generar sugerencias REALES con Google Places para día vacío
                    suggested_places = await generate_smart_suggestions_for_day(
                        places_service, center_lat, center_lon, day_date, request, day_counter, used_place_ids_global
                    )
                    
                    # Crear día con sugerencias
                    empty_day_formatted = {
                        "day": len(itinerary_days) + 1,
                        "date": day_date,
                        "places": suggested_places,
                        "transfers": [],
                        "total_places": len(suggested_places),
                        "total_transfers": 0,
                        "total_time": "3.5h",
                        "walking_time": "0min",
                        "transport_time": "0min", 
                        "free_time": "5.5h",
                        "suggested_day": True,
                        "suggestion_reason": f"Día libre detectado - sugerencias automáticas generadas"
                    }
                    
                    # Evitar duplicar si por alguna razón ya existe
                    if not any(d.get('date') == day_date for d in itinerary_days):
                        itinerary_days.append(empty_day_formatted)
                        day_counter += 1
                        logger.info(f"✅ Día vacío {day_date} creado con {len(suggested_places)} sugerencias")
                    else:
                        logger.info(f"⚠️ Día {day_date} ya existía en el itinerario, se omitió creación")
                
                # Ordenar días por fecha para mantener secuencia correcta
                itinerary_days.sort(key=lambda x: x.get("date", ""))
                
                # Actualizar recomendaciones
                if empty_days:
                    base_recommendations.extend([
                        f"🎯 {len(empty_days)} días libres detectados - sugerencias automáticas generadas",
                        "💡 Las sugerencias aparecen marcadas como 'suggested: true'",
                        "🔄 Puedes reemplazar las sugerencias con tus propios lugares"
                    ])
                    
                logger.info(f"🎉 Sugerencias automáticas completadas: {len(empty_days)} días procesados")
                
            except Exception as e:
                logger.error(f"❌ Error generando sugerencias automáticas: {e}")
                base_recommendations.append(
                    "⚠️ Error generando sugerencias automáticas para días libres. "
                    "Contacta soporte si persiste el problema."
                )
        
        return ItineraryResponse(
            itinerary=itinerary_days,
            optimization_metrics={
                "total_places": total_activities,
                "total_days": len(itinerary_days),
                "optimization_mode": "multimodal_hybrid_v31",
                "efficiency_score": optimization_metrics.get('efficiency_score', 0.95),
                "total_travel_time_minutes": total_travel_minutes,
                "total_transport_time_minutes": total_transport_minutes,
                "total_walking_time_minutes": total_walking_minutes,
                "processing_time_seconds": round(duration, 2),
                "multimodal_router_stats": router_stats,
                "clusters_generated": optimization_metrics.get('clusters_generated', 0),
                "intercity_transfers_detected": optimization_metrics.get('long_transfers_detected', 0),
                "cache_performance": optimization_metrics.get('cache_performance', {})
            },
            recommendations=base_recommendations
        )
        
    except Exception as e:
        logger.error(f"💥 Error en itinerario multi-modal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generando itinerario multi-modal: {str(e)}")

# ===== FUNCIONES AUXILIARES PARA SUGERENCIAS INTELIGENTES =====

async def generate_smart_suggestions_for_day(places_service, center_lat, center_lon, day_date, request, day_number=1, used_place_ids=None):
    """
    🎯 Generar sugerencias inteligentes usando Google Places API real con VARIEDAD
    
    Features:
    - ✨ Google Places API real para lugares auténticos
    - 🏷️ Personalización basada en preferencias del usuario  
    - 📍 Lugares populares y bien valorados de la zona
    - ⭐ Ratings, reviews y información detallada
    - 🔄 Variedad por día - evita repetir lugares
    """
    suggested_places = []
    
    # Cache global de lugares ya usados para evitar repeticiones
    if used_place_ids is None:
        used_place_ids = set()
    
    try:
        # Obtener preferencias del usuario
        user_preferences = getattr(request, 'preferences', {})
        interests = user_preferences.get('interests', ['tourist_attraction', 'restaurant', 'museum'])
        budget_level = user_preferences.get('budget', 'medium')
        
        # Mapear presupuesto a filtros de precio de Google Places
        price_levels = {
            'low': [1, 2],      # $ y $$
            'medium': [2, 3],   # $$ y $$$  
            'high': [3, 4],     # $$$ y $$$$
            'any': [1, 2, 3, 4] # Todos los niveles
        }
        
        # 🎯 Categorías organizadas por PRIORIDAD: atracciones primero, dining después
        # Nunca incluir hotel/lodging/accommodation como actividad
        EXCLUDED_TYPES = {'hotel', 'lodging', 'accommodation', 'motel', 'hostel', 'resort'}

        # Tier 1: Lugares representativos (landmarks, cultura, naturaleza)
        primary_categories = [
            'tourist_attraction', 'museum', 'park', 'art_gallery',
            'church', 'monument', 'beach', 'zoo', 'viewpoint',
            'natural_feature', 'point_of_interest', 'amusement_park'
        ]

        # Tier 2: Dining y comercio (solo si no hay suficientes del tier 1)
        secondary_categories = [
            'restaurant', 'cafe', 'shopping_mall', 'store',
            'night_club', 'bar', 'bakery'
        ]

        # Filtrar por preferencias del usuario manteniendo la prioridad
        user_primary = []
        user_secondary = []

        if 'tourist_attraction' in interests or 'culture' in str(user_preferences):
            user_primary.extend(['tourist_attraction', 'museum', 'art_gallery', 'monument', 'church'])
        if 'nature' in interests or 'nature' in str(user_preferences):
            user_primary.extend(['park', 'natural_feature', 'zoo', 'beach', 'viewpoint'])
        if 'restaurant' in interests or 'food' in str(user_preferences):
            user_secondary.extend(['restaurant', 'cafe', 'bakery'])
        if 'shopping' in interests:
            user_secondary.extend(['shopping_mall', 'store'])
        if 'nightlife' in interests:
            user_secondary.extend(['night_club', 'bar'])

        # Combinar: primero primary, luego secondary (sin duplicados)
        seen = set()
        all_categories = []
        for cat in (user_primary or primary_categories) + (user_secondary or secondary_categories):
            if cat not in seen and cat not in EXCLUDED_TYPES:
                all_categories.append(cat)
                seen.add(cat)

        # Fallback: siempre asegurar atracciones primero
        if not all_categories:
            all_categories = primary_categories + secondary_categories

        # 🔄 Rotar categorías por día, priorizando siempre atracciones
        # Cada día toma 3 categorías: al menos 2 del tier primario
        day_offset = (day_number - 1) * 2
        search_categories = []

        # Separar categorías disponibles en primary vs secondary
        available_primary = [c for c in all_categories if c in set(primary_categories)]
        available_secondary = [c for c in all_categories if c in set(secondary_categories)]

        # Tomar 2 del primary (rotando) + 1 del secondary (rotando)
        for i in range(2):
            if available_primary:
                idx = (day_offset + i) % len(available_primary)
                search_categories.append(available_primary[idx])
        if available_secondary:
            idx = (day_number - 1) % len(available_secondary)
            search_categories.append(available_secondary[idx])
        elif available_primary:
            # Si no hay secondary, tomar otro primary
            idx = (day_offset + 2) % len(available_primary)
            if available_primary[idx] not in search_categories:
                search_categories.append(available_primary[idx])
        
        logger.info(f"🎲 Día {day_number}: Categorías rotadas = {search_categories}")
        
        # 📍 Variar también el radio de búsqueda según el día para más diversidad
        base_radius = 8000  # 8km base
        radius_variation = (day_number % 3) * 2000  # +0km, +2km, +4km según día
        search_radius = base_radius + radius_variation
        
        order_counter = 1
        logger.info(f"🔍 Generando sugerencias para categorías: {search_categories}")
        
        # 📍 Buscar lugares reales por cada categoría de interés
        for category in search_categories[:3]:  # Max 3 categorías por día (limitado)
            try:
                logger.info(f"🔍 Buscando {category} cerca de {center_lat:.4f}, {center_lon:.4f}")
                
                # 🎯 Parámetros de búsqueda optimizados para calidad con variedad
                search_params = {
                    "query": f"best {category.replace('_', ' ')} popular top rated must visit",
                    "location": f"{center_lat},{center_lon}",
                    "radius": search_radius,  # Radio variable según día
                    "place_type": category,
                    "limit": 3,  # Buscar 3 para tener opciones y evitar repetidos
                    "min_rating": 4.0  # Solo lugares bien valorados
                }
                
                # 💰 Agregar filtro de precio según presupuesto del usuario
                if budget_level in price_levels:
                    search_params["price_levels"] = price_levels[budget_level]
                
                # 🌟 Buscar lugares reales con Google Places
                logger.info(f"🔍 Llamando Google Places API: lat={center_lat:.4f}, lon={center_lon:.4f}, types={[category]}, radius={search_params['radius']}")
                real_suggestions = await places_service.search_nearby(
                    lat=center_lat,
                    lon=center_lon,
                    types=[category],
                    radius_m=search_params["radius"],
                    limit=search_params["limit"]
                )
                logger.info(f"📊 Google Places API devolvió: {len(real_suggestions)} resultados para {category}")
                
                # 🚫 Filtrar lugares ya usados y hoteles/alojamientos
                filtered_suggestions = []
                for suggestion in real_suggestions:
                    place_id = suggestion.get('place_id', '')
                    suggestion_name = suggestion.get('name', '').lower()
                    suggestion_types = [t.lower() for t in suggestion.get('types', [])]

                    # Excluir hoteles y alojamientos
                    if any(t in EXCLUDED_TYPES for t in suggestion_types):
                        logger.info(f"🚫 Excluido (alojamiento): {suggestion.get('name', '')} tipos={suggestion_types}")
                        continue

                    # Heuristic: excluir por nombre si parece hotel
                    hotel_keywords = ['hotel', 'hostel', 'hostal', 'motel', 'resort', 'inn ', 'lodge', 'apart hotel']
                    if any(kw in suggestion_name for kw in hotel_keywords):
                        logger.info(f"🚫 Excluido (nombre de alojamiento): {suggestion.get('name', '')}")
                        continue

                    # Evitar lugares ya usados por ID o nombre similar
                    if place_id and place_id not in used_place_ids:
                        # También evitar nombres muy similares
                        name_already_used = any(
                            suggestion_name in used_name.lower() or used_name.lower() in suggestion_name
                            for used_name in [p.get('name', '') for p in suggested_places]
                        )

                        if not name_already_used:
                            filtered_suggestions.append(suggestion)
                            used_place_ids.add(place_id)
                            break  # Solo tomar 1 por categoría para mantener límite
                
                # Procesar solo las sugerencias filtradas
                for suggestion in filtered_suggestions:
                    # ⏰ Calcular tiempo estimado basado en tipo de lugar
                    duration_map = {
                        'restaurant': ("1.5h", "12:00-14:00"),
                        'cafe': ("1.0h", "15:00-16:00"), 
                        'museum': ("2.5h", "10:00-12:30"),
                        'art_gallery': ("2.0h", "10:30-12:30"),
                        'park': ("1.5h", "09:00-10:30"),
                        'natural_feature': ("2.0h", "09:00-11:00"),
                        'shopping_mall': ("2.0h", "14:00-16:00"),
                        'store': ("1.0h", "15:00-16:00"),
                        'tourist_attraction': ("2.5h", "10:00-12:30"),
                        'point_of_interest': ("2.0h", "10:00-12:00"),
                        'night_club': ("3.0h", "21:00-00:00"),
                        'bar': ("2.0h", "19:00-21:00")
                    }
                    
                    estimated_duration, best_time = duration_map.get(category, ("2.0h", "10:00-12:00"))
                    
                    # 🎯 Crear lugar sugerido adaptable (Google Places real o sintético)
                    is_synthetic = suggestion.get("synthetic", False)
                    place_name = suggestion.get("name", f"Lugar {category}")
                    
                    # Descripción inteligente basada en el origen
                    if is_synthetic:
                        description = f"📍 Sugerencia local para {category.replace('_', ' ')} - {place_name}"
                        reason_text = f"🎯 Sugerencia basada en tus intereses para {day_date}"
                        verified_status = False
                    else:
                        description = f"✨ Recomendado por Google Places - {place_name} es un destino destacado en el área"
                        reason_text = f"🎯 Sugerencia inteligente basada en tus intereses ({', '.join(interests)}) para {day_date}"
                        verified_status = True
                    
                    suggested_place = {
                        "id": f"suggested-{day_date}-{order_counter}",
                        "name": place_name,
                        "category": category,
                        "rating": suggestion.get("rating", 4.0),
                        "image": suggestion.get("photo_url", suggestion.get("photo", "")),
                        "description": description,
                        "estimated_time": estimated_duration,
                        "priority": 3 + (order_counter % 2),
                        "lat": suggestion.get("lat", center_lat),
                        "lng": suggestion.get("lon", suggestion.get("lng", center_lon)),
                        "recommended_duration": estimated_duration,
                        "best_time": best_time,
                        "order": order_counter,
                        "is_intercity": False,
                        "suggested": True,
                        "suggestion_reason": reason_text,
                        
                        # 🌟 Campos adaptativos según el origen
                        "address": suggestion.get("address", "Dirección no disponible"),
                        "phone": suggestion.get("phone", ""),
                        "website": suggestion.get("website", ""),
                        "reviews_count": suggestion.get("user_ratings_total", suggestion.get("reviews_count", 0)),
                        "place_id": suggestion.get("place_id", ""),
                        "price_level": suggestion.get("price_level", 0),
                        "types": suggestion.get("types", [category]),
                        "opening_hours": suggestion.get("opening_hours", {}),
                        "google_maps_url": f"https://www.google.com/maps/search/{place_name.replace(' ', '+')}/@{suggestion.get('lat', center_lat)},{suggestion.get('lon', suggestion.get('lng', center_lon))}",
                        
                        # 📊 Información de calidad
                        "google_places_verified": verified_status,
                        "synthetic": is_synthetic,
                        "popularity_score": suggestion.get("rating", 4.0),
                        "budget_match": budget_level,
                        "interest_match": [interest for interest in interests if interest in category or category in interest],
                        "eta_minutes": suggestion.get("eta_minutes", 5)
                    }
                    
                    suggested_places.append(suggested_place)
                    order_counter += 1
                    
                    logger.info(f"✅ Sugerencia real: {suggested_place['name']} ({suggested_place['rating']}⭐, {suggested_place['reviews_count']} reviews)")
                    
                # Limitar a 3 lugares por día para no sobrecargar el itinerario
                if len(suggested_places) >= 3:
                    break
                    
            except Exception as e:
                logger.warning(f"⚠️ Error buscando {category}: {e}")
                
                # 🔄 Fallback con sugerencia genérica de calidad si Google Places falla
                fallback_suggestion = {
                    "id": f"suggested-{day_date}-fallback-{order_counter}",
                    "name": f"Explorar {category.replace('_', ' ')} local",
                    "category": category,
                    "rating": 4.0,
                    "image": "",
                    "description": f"Sugerencia local para {category.replace('_', ' ')}",
                    "estimated_time": "2.0h",
                    "priority": 3,
                    "lat": center_lat,
                    "lng": center_lon,
                    "recommended_duration": "2.0h",
                    "best_time": "10:00-12:00",
                    "order": order_counter,
                    "is_intercity": False,
                    "suggested": True,
                    "suggestion_reason": f"Sugerencia automática para {day_date}",
                    "google_places_verified": False
                }
                suggested_places.append(fallback_suggestion)
                order_counter += 1
        
        # 🛡️ Asegurar exactamente 3 sugerencias por día con VARIEDAD
        while len(suggested_places) < 3:
            logger.warning(f"⚠️ Solo {len(suggested_places)} sugerencias encontradas para {day_date}, agregando fallback")
            
            # 🎲 Crear sugerencias fallback VARIADAS por día - priorizar atracciones
            fallback_options = [
                # Día 1: landmarks + cultura
                [("tourist_attraction", "Monumento principal"), ("museum", "Museo de historia local"), ("park", "Parque central")],
                # Día 2: naturaleza + mirador
                [("viewpoint", "Mirador panorámico"), ("beach", "Playa o costanera local"), ("art_gallery", "Galería de arte")],
                # Día 3: cultura + punto de interés
                [("church", "Catedral o iglesia histórica"), ("point_of_interest", "Plaza histórica"), ("museum", "Centro cultural")],
                # Día 4+: naturaleza + atracciones (ciclar)
                [("park", "Jardín botánico"), ("tourist_attraction", "Sitio emblemático"), ("natural_feature", "Reserva natural")]
            ]
            
            # Seleccionar opciones basadas en día para garantizar variedad
            day_cycle = (day_number - 1) % len(fallback_options)
            day_fallbacks = fallback_options[day_cycle]
            
            current_count = len(suggested_places)
            for i in range(3 - current_count):
                if i < len(day_fallbacks):
                    category, name = day_fallbacks[i]
                    
                    # Generar coordenadas ligeramente diferentes para cada sugerencia
                    lat_offset = (i + day_number * 0.5) * 0.001  # Variación por día y posición
                    lon_offset = (i + day_number * 0.3) * 0.001
                    
                    fallback_place = {
                        "id": f"suggested-{day_date}-varied-{i+1}",
                        "name": name,
                        "category": category,
                        "rating": 4.1 + (i * 0.1),  # Ratings ligeramente diferentes 
                        "image": "",
                        "description": f"Sugerencia día {day_number} - {name}",
                        "estimated_time": "1.5h",
                        "priority": 4,
                        "lat": center_lat + lat_offset,
                        "lng": center_lon + lon_offset,
                        "recommended_duration": "1.5h",
                        "best_time": "12:30-14:00",
                        "order": current_count + i + 1,
                        "is_intercity": False,
                        "suggested": True,
                        "suggestion_reason": f"Sugerencia variada para {day_date} (día {day_number})",
                        "google_places_verified": False,
                        "synthetic": True
                    }
                    suggested_places.append(fallback_place)
        
        logger.info(f"🎉 Generadas {len(suggested_places)} sugerencias inteligentes para {day_date}")
        return suggested_places
        
    except Exception as e:
        logger.error(f"❌ Error generando sugerencias inteligentes para {day_date}: {e}")
        
        # 🔄 Fallback completo si todo falla - exactamente 3 sugerencias
        fallback_suggestions = [
            {
                "id": f"suggested-{day_date}-emergency-1",
                "name": "Explorar área local",
                "category": "tourist_attraction", 
                "rating": 4.0,
                "image": "",
                "description": f"Sugerencia de emergencia para {day_date}",
                "estimated_time": "2.0h",
                "priority": 3,
                "lat": center_lat,
                "lng": center_lon,
                "recommended_duration": "2.0h",
                "best_time": "10:00-12:00",
                "order": 1,
                "is_intercity": False,
                "suggested": True,
                "suggestion_reason": f"Sugerencia automática para {day_date}",
                "google_places_verified": False,
                "synthetic": True
            },
            {
                "id": f"suggested-{day_date}-emergency-2",
                "name": "Museo o centro cultural",
                "category": "museum",
                "rating": 4.0,
                "image": "",
                "description": f"Museo o centro cultural sugerido para {day_date}",
                "estimated_time": "2.0h",
                "priority": 3,
                "lat": center_lat + 0.001,
                "lng": center_lon + 0.001,
                "recommended_duration": "2.0h",
                "best_time": "10:30-12:30",
                "order": 2,
                "is_intercity": False,
                "suggested": True,
                "suggestion_reason": f"Sugerencia automática para {day_date}",
                "google_places_verified": False,
                "synthetic": True
            },
            {
                "id": f"suggested-{day_date}-emergency-3",
                "name": "Actividad recreativa",
                "category": "park",
                "rating": 4.0,
                "image": "",
                "description": f"Parque o actividad al aire libre para {day_date}",
                "estimated_time": "1.5h",
                "priority": 3,
                "lat": center_lat + 0.002,
                "lng": center_lon + 0.002,
                "recommended_duration": "1.5h",
                "best_time": "15:00-17:00",
                "order": 3,
                "is_intercity": False,
                "suggested": True,
                "suggestion_reason": f"Sugerencia automática para {day_date}",
                "google_places_verified": False,
                "synthetic": True
            }
        ]
        
        return fallback_suggestions


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=getattr(settings, 'API_HOST', '0.0.0.0'),
        port=getattr(settings, 'API_PORT', 8000),
        reload=getattr(settings, 'DEBUG', True)
    )
