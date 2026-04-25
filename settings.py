import os
from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()


def _safe_int(env_name: str, default: int) -> int:
    """Safely parse int from env var, returning default on failure."""
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def _safe_float(env_name: str, default: float) -> float:
    """Safely parse float from env var, returning default on failure."""
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        return default


def _safe_bool(env_name: str, default: bool) -> bool:
    """Safely parse bool from env var."""
    return os.getenv(env_name, str(default)).lower() in ("true", "1", "yes")


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "sqlite:///goveling.db"

    # ML Model
    MODEL_PATH: str = "models/duration_model.pkl"
    RETRAIN_THRESHOLD_DAYS: int = 30

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = _safe_int("API_PORT", 8000)
    DEBUG: bool = _safe_bool("DEBUG", False)
    API_KEY: Optional[str] = os.getenv("API_KEY")

    # Performance
    ENABLE_CACHE: bool = _safe_bool("ENABLE_CACHE", True)
    CACHE_TTL_SECONDS: int = _safe_int("CACHE_TTL_SECONDS", 300)
    MAX_CONCURRENT_REQUESTS: int = _safe_int("MAX_CONCURRENT_REQUESTS", 3)
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600
    
    # External APIs
    GOOGLE_MAPS_API_KEY: Optional[str] = None
    GOOGLE_PLACES_API_KEY: Optional[str] = os.getenv("GOOGLE_PLACES_API_KEY")
    OPENAI_API_KEY: Optional[str] = None
    ENABLE_REAL_PLACES: bool = os.getenv("ENABLE_REAL_PLACES", "true").lower() == "true"

    # Supabase — used by DwellStatsLookup (services/dwell_stats_service.py) to
    # read learned per-place / per-category dwell-time percentiles. Both vars
    # optional; lookup degrades to hardcoded duration_map when absent so local
    # dev / tests work without Supabase access.
    SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY: Optional[str] = os.getenv("SUPABASE_ANON_KEY")
    
    # Free Routing APIs (alternativas gratuitas a Google Directions)
    OPENROUTE_API_KEY: Optional[str] = os.getenv('OPENROUTE_API_KEY', None)  # Obtener clave gratuita en openrouteservice.org
    FREE_ROUTING_TIMEOUT: int = 8  # segundos
    ROUTING_FALLBACK_BUFFER_URBAN: float = 1.4  # 40% buffer en ciudades
    ROUTING_FALLBACK_BUFFER_RURAL: float = 1.2  # 20% buffer en zonas rurales
    
    # Business Logic
    MAX_DAILY_ACTIVITIES: int = 8
    MAX_WALKING_DISTANCE_KM: float = 15.0
    DEFAULT_ACTIVITY_BUFFER_MIN: int = 15
    
    # Horario comercial universal (MVP)
    BUSINESS_START_H: int = 9    # 09:00
    BUSINESS_END_H: int = 18     # 18:00

    # Velocidades aproximadas (para traslados)
    CITY_SPEED_KMH_WALK: float = 4.5   # caminata urbana
    CITY_SPEED_KMH_DRIVE: float = 22.0  # conducción urbana
    CITY_SPEED_KMH_BIKE: float = 15.0   # ciclismo urbana
    CITY_SPEED_KMH_TRANSIT: float = 25.0  # transporte público
    MIN_TRAVEL_MIN: int = 8     # mínimo realista por traslado
    
    # Clustering geográfico y detección de traslados largos
    CLUSTER_EPS_KM_URBAN: float = 8.0   # Radio clustering en zonas urbanas densas
    CLUSTER_EPS_KM_RURAL: float = 15.0  # Radio clustering en zonas rurales/turisticas
    CLUSTER_MIN_SAMPLES: int = 1        # Mínimo de lugares para formar un cluster
    WALK_MAX_KM: float = 2.0           # Máximo para caminar (>2km forzar auto/bus)
    INTERCITY_THRESHOLD_KM_URBAN: float = 25.0   # Umbral intercity urbano
    INTERCITY_THRESHOLD_KM_RURAL: float = 40.0   # Umbral intercity rural
    LONG_TRANSFER_MIN: int = 120       # Minutos para considerar un traslado "largo"
    
    # Velocidades para fallback cuando Google Directions falla
    WALK_KMH: float = 4.5              # Velocidad caminando
    DRIVE_KMH: float = 50.0            # Velocidad en auto (interurbano)
    TRANSIT_KMH: float = 35.0          # Velocidad transporte público
    AIR_SPEED_KMPH: float = 750.0      # Velocidad promedio vuelo comercial (incluyendo tiempo aeropuerto)
    AIR_BUFFERS_MIN: int = 90          # Buffers aeropuerto (check-in, security, boarding, etc.)
    
    # Políticas de transporte por distancia
    WALK_THRESHOLD_KM: float = 2.0     # <= 2km: caminar OK
    DRIVE_THRESHOLD_KM: float = 15.0   # > 15km: driving recomendado
    FLIGHT_THRESHOLD_KM: float = 1000.0 # > 1000km: vuelo recomendado
    TRANSIT_AVAILABLE: bool = True      # Si hay transporte público disponible
    
    # Ventanas horarias por tipo de lugar
    RESTAURANT_LUNCH_START: int = 12    # 12:00
    RESTAURANT_LUNCH_END: int = 15      # 15:00
    RESTAURANT_DINNER_START: int = 19   # 19:00
    RESTAURANT_DINNER_END: int = 22     # 22:00
    MUSEUM_PREFERRED_START: int = 10    # 10:00
    MUSEUM_PREFERRED_END: int = 17      # 17:00
    SHOPPING_PREFERRED_START: int = 10  # 10:00
    SHOPPING_PREFERRED_END: int = 20    # 20:00
    
    # Estrategias de empaquetado
    DEFAULT_PACKING_STRATEGY: str = "balanced"  # "compact" | "balanced" | "cluster_first"
    MIN_ACTIVITIES_PER_DAY: int = 2
    MAX_ACTIVITIES_PER_DAY: int = 6
    TARGET_MINUTES_PER_DAY: int = 300   # 5 horas de actividades por día
    
    # Sugerencias para días libres
    FREE_DAY_SUGGESTIONS_RADIUS_M: int = 3000
    FREE_DAY_SUGGESTIONS_LIMIT: int = 3  # Reducido de 6 a 3 para mejor UX
    
    # ========================================================================
    # 🧠 CITY2GRAPH CONFIGURATION - FASE 1 (FEATURE FLAGS)
    # ========================================================================
    
    # Master switch
    ENABLE_CITY2GRAPH: bool = _safe_bool("ENABLE_CITY2GRAPH", False)

    # Criterios de activacion
    CITY2GRAPH_MIN_PLACES: int = _safe_int("CITY2GRAPH_MIN_PLACES", 8)
    CITY2GRAPH_MIN_DAYS: int = _safe_int("CITY2GRAPH_MIN_DAYS", 3)
    CITY2GRAPH_COMPLEXITY_THRESHOLD: float = _safe_float("CITY2GRAPH_COMPLEXITY_THRESHOLD", 5.0)

    # Control geografico
    CITY2GRAPH_CITIES: str = os.getenv("CITY2GRAPH_CITIES", "")
    CITY2GRAPH_EXCLUDE_CITIES: str = os.getenv("CITY2GRAPH_EXCLUDE_CITIES", "")

    # Performance y reliability
    CITY2GRAPH_TIMEOUT_S: int = _safe_int("CITY2GRAPH_TIMEOUT_S", 30)
    CITY2GRAPH_FALLBACK_ENABLED: bool = _safe_bool("CITY2GRAPH_FALLBACK_ENABLED", True)
    CITY2GRAPH_MAX_CONCURRENT: int = _safe_int("CITY2GRAPH_MAX_CONCURRENT", 1)

    # Circuit Breaker
    CITY2GRAPH_FAILURE_THRESHOLD: int = _safe_int("CITY2GRAPH_FAILURE_THRESHOLD", 5)
    CITY2GRAPH_RECOVERY_TIMEOUT: int = _safe_int("CITY2GRAPH_RECOVERY_TIMEOUT", 300)

    # A/B Testing
    CITY2GRAPH_USER_PERCENTAGE: int = _safe_int("CITY2GRAPH_USER_PERCENTAGE", 0)
    CITY2GRAPH_TRACK_DECISIONS: bool = _safe_bool("CITY2GRAPH_TRACK_DECISIONS", True)

    # Deteccion avanzada
    CITY2GRAPH_GEO_SPREAD_THRESHOLD_KM: float = _safe_float("CITY2GRAPH_GEO_SPREAD_THRESHOLD_KM", 50.0)
    CITY2GRAPH_SEMANTIC_TYPES_THRESHOLD: int = _safe_int("CITY2GRAPH_SEMANTIC_TYPES_THRESHOLD", 3)
    
    # ========================================================================
    # 🧮 OR-TOOLS CONFIGURATION - FASE 2 (POST-BENCHMARK INTEGRATION)
    # ========================================================================
    
    # Master switch
    ENABLE_ORTOOLS: bool = _safe_bool("ENABLE_ORTOOLS", False)

    # Criterios de activacion
    ORTOOLS_MIN_PLACES: int = _safe_int("ORTOOLS_MIN_PLACES", 4)
    ORTOOLS_MIN_DAYS: int = _safe_int("ORTOOLS_MIN_DAYS", 1)
    ORTOOLS_MAX_PLACES: int = _safe_int("ORTOOLS_MAX_PLACES", 50)
    ORTOOLS_MAX_DISTANCE_KM: int = _safe_int("ORTOOLS_MAX_DISTANCE_KM", 500)

    # Performance
    ORTOOLS_TIMEOUT_S: int = _safe_int("ORTOOLS_TIMEOUT_S", 10)
    ORTOOLS_SLOW_THRESHOLD_MS: int = _safe_int("ORTOOLS_SLOW_THRESHOLD_MS", 5000)
    ORTOOLS_EXPECTED_EXEC_TIME_MS: int = 2000

    # Control geografico
    ORTOOLS_CITIES: str = os.getenv("ORTOOLS_CITIES", "santiago,valparaiso,antofagasta,la_serena,concepcion,temuco,iquique,calama")
    ORTOOLS_EXCLUDE_CITIES: str = os.getenv("ORTOOLS_EXCLUDE_CITIES", "")

    # Circuit Breaker
    ORTOOLS_FAILURE_THRESHOLD: int = _safe_int("ORTOOLS_FAILURE_THRESHOLD", 3)
    ORTOOLS_RECOVERY_TIMEOUT: int = _safe_int("ORTOOLS_RECOVERY_TIMEOUT", 60)
    ORTOOLS_HEALTH_CHECK_TTL: int = _safe_int("ORTOOLS_HEALTH_CHECK_TTL", 300)

    # A/B Testing
    ORTOOLS_USER_PERCENTAGE: int = _safe_int("ORTOOLS_USER_PERCENTAGE", 50)
    ORTOOLS_TRACK_PERFORMANCE: bool = _safe_bool("ORTOOLS_TRACK_PERFORMANCE", True)

    # Advanced Constraints
    ORTOOLS_ENABLE_TIME_WINDOWS: bool = _safe_bool("ORTOOLS_ENABLE_TIME_WINDOWS", True)
    ORTOOLS_ENABLE_VEHICLE_ROUTING: bool = _safe_bool("ORTOOLS_ENABLE_VEHICLE_ROUTING", True)
    ORTOOLS_ENABLE_ADVANCED_CONSTRAINTS: bool = _safe_bool("ORTOOLS_ENABLE_ADVANCED_CONSTRAINTS", True)
    ORTOOLS_OPTIMIZATION_TARGET: str = os.getenv("ORTOOLS_OPTIMIZATION_TARGET", "minimize_travel_time")

    # Performance Optimization
    ORTOOLS_ENABLE_PARALLEL_OPTIMIZATION: bool = _safe_bool("ORTOOLS_ENABLE_PARALLEL_OPTIMIZATION", True)
    ORTOOLS_CACHE_DISTANCE_MATRIX: bool = _safe_bool("ORTOOLS_CACHE_DISTANCE_MATRIX", True)
    ORTOOLS_DISTANCE_CACHE_TTL: int = _safe_int("ORTOOLS_DISTANCE_CACHE_TTL", 3600)
    ORTOOLS_MAX_PARALLEL_REQUESTS: int = _safe_int("ORTOOLS_MAX_PARALLEL_REQUESTS", 3)

    # Multi-City Integration
    ORTOOLS_ENABLE_MULTI_CITY: bool = _safe_bool("ORTOOLS_ENABLE_MULTI_CITY", True)
    ORTOOLS_MULTI_CITY_THRESHOLD_KM: int = _safe_int("ORTOOLS_MULTI_CITY_THRESHOLD_KM", 100)
    ORTOOLS_ACCOMMODATE_MULTI_CITY: bool = _safe_bool("ORTOOLS_ACCOMMODATE_MULTI_CITY", True)

    # Fallback strategy
    ORTOOLS_FALLBACK_TO_LEGACY: bool = _safe_bool("ORTOOLS_FALLBACK_TO_LEGACY", True)
    ORTOOLS_FALLBACK_ON_SLOW: bool = _safe_bool("ORTOOLS_FALLBACK_ON_SLOW", False)

    # Monitoreo y alertas
    ORTOOLS_LOG_PERFORMANCE: bool = _safe_bool("ORTOOLS_LOG_PERFORMANCE", True)
    ORTOOLS_ALERT_ON_DEGRADATION: bool = _safe_bool("ORTOOLS_ALERT_ON_DEGRADATION", True)

    # Benchmark validation
    ORTOOLS_VALIDATE_VS_BENCHMARKS: bool = _safe_bool("ORTOOLS_VALIDATE_VS_BENCHMARKS", True)
    ORTOOLS_BENCHMARK_SUCCESS_RATE_THRESHOLD: float = _safe_float("ORTOOLS_BENCHMARK_SUCCESS_RATE_THRESHOLD", 0.95)
    
    class Config:
        env_file = ".env"

settings = Settings()
