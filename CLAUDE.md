# goveling-ml — FastAPI ML Itinerary Optimizer

## Overview
AI-powered itinerary optimization service using Google OR-Tools for TSP/VRP problems. Generates optimized multi-day travel itineraries with real routing data. Deployed on Render.

- **Framework**: FastAPI (Python 3.11+)
- **Optimization**: Google OR-Tools 9.8+ (TSP/VRP), scikit-learn, networkx
- **Production**: `https://goveling-ml.onrender.com`
- **Port**: 8000

## Project Structure
```
goveling-ml/
├── api.py                    # Main FastAPI app (3500+ lines) — all core endpoints
├── settings.py               # Central configuration (env vars, feature flags, thresholds)
├── requirements.txt
├── routers/
│   ├── routing.py            # /route/* — drive/walk/bike/compare endpoints
│   ├── monitoring.py         # /api/v4/monitoring/* — health, dashboard, alerts
│   └── semantic.py           # /semantic/* — City2Graph semantic analysis
├── models/
│   ├── schemas.py            # Pydantic models (Place, ItineraryRequest/Response, etc.)
│   └── route_schemas.py      # Route-specific schemas
├── middleware/
│   ├── auth.py               # Optional API key middleware
│   └── rate_limit.py         # Rate limiting
├── services/                 # ~30 files — core business logic
│   ├── city2graph_ortools_service.py   # OR-Tools TSP/VRP optimization engine
│   ├── ortools_distance_cache.py       # Distance matrix caching (OSRM)
│   ├── ortools_parallel_optimizer.py   # Multi-core parallel processing
│   ├── ortools_advanced_constraints.py # Time windows, vehicle routing
│   ├── ortools_monitoring.py           # Real-time metrics and alerting
│   ├── google_places_service.py        # Google Places API
│   ├── hotel_recommender.py            # Hotel recommendation engine
│   ├── chile_multimodal_router.py      # Multi-modal routing (Chile)
│   ├── multi_city_optimizer.py         # Multi-city itinerary optimization
│   └── city_clustering_service.py      # Geographic clustering detection
├── utils/
│   ├── hybrid_optimizer_v31.py  # Main optimization coordinator (decision engine)
│   ├── ortools_decision_engine.py # Smart optimizer selection
│   ├── geo_utils.py              # Haversine distance, geographic calcs
│   ├── google_maps_client.py     # Google Maps API client
│   └── hybrid_routing_service.py # Routing abstraction layer
└── cache/                    # Local caching directories
```

## Key Endpoints

### Itinerary Generation (consumed by RN app)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v2/itinerary/generate-hybrid` | **Primary** — OR-Tools optimized itinerary |
| POST | `/api/v1/itinerary/generate-hybrid` | **Fallback** — legacy optimization |

### Routing
| POST | `/route/drive` | Driving route |
| POST | `/route/walk` | Walking route |
| POST | `/route/bike` | Biking route |
| POST | `/route/compare` | Compare all modes with recommendations |

### Monitoring
| GET | `/api/v4/monitoring/health` | Health check with scoring |
| GET | `/api/v4/monitoring/dashboard` | Comprehensive metrics |
| GET | `/api/v4/monitoring/alerts` | Active alerts (24h) |
| GET | `/api/v4/monitoring/benchmark` | OR-Tools vs Legacy comparison |

### Other
| GET | `/health` | Basic health check |
| POST | `/semantic/analyze-real` | Semantic analysis with real OSM data |

## Optimization Architecture
```
Request → HybridOptimizerV31 (decision engine)
              ↓
         OR-Tools Professional → Distance Cache (OSRM)
              ↓ (fallback)        ↓
         City2Graph Legacy    Distance Matrix
              ↓ (fallback)
         Greedy Algorithm (local, in RN app)
```

- **Circuit breaker pattern** for graceful failure recovery
- **A/B testing** via `ORTOOLS_USER_PERCENTAGE` (currently 50%)
- **8 Chilean cities** enabled: Santiago, Valparaíso, Antofagasta, La Serena, Concepción, Temuco, Iquique, Calama

## Environment Variables
```
# Feature Flags
ENABLE_ORTOOLS=true/false
ORTOOLS_USER_PERCENTAGE=0-100
ORTOOLS_CITIES=Santiago,Valparaiso,...
ORTOOLS_TIMEOUT_SECONDS=30
ENABLE_CITY2GRAPH=false

# API Keys
GOOGLE_PLACES_API_KEY=
OPENROUTE_API_KEY=
API_KEY=                    # Optional auth

# Infrastructure
CORS_ORIGINS=*
DEBUG=true/false
```

## Common Commands
```bash
# Development
uvicorn api:app --reload --port 8000
python api.py

# Testing
python test_endpoints.py
python test_chile_multicity.py

# Deployment
./deploy_render.sh
./clean_for_production.sh
```

## Business Logic Thresholds (settings.py)
- Max daily activities: 8
- Walking limit: 15km
- Speed: walk 4.5 km/h, drive 22 km/h, bike 15 km/h, transit 25 km/h
- Min travel time: 8 min (realistic)
- OR-Tools: max 50 places, 500km distance, 10s timeout
- Cache TTL: 300s (requests), 3600s (distance matrix)

## Consumed By
- `Goveling-rn2025/src/lib/aiRoutesService.ts` — main consumer
- Fallback chain: v2 → v1 → local greedy (in RN app)
- No auth required (rate-limited server-side)

## API Contract
```python
# Request (ItineraryRequest):
{ trip_id, start_date, end_date, mode?, daily_window?, accommodations?, places: [Place] }

# Place:
{ id, name, lat, lng, priority?, duration_min?, category?, open_hours? }

# Response:
{ ok: bool, days: [DayPlan], version: str, error?: str }

# DayPlan:
{ date, places: [{id, name, lat, lng, eta?, etd?}], metrics?: {distance_m?, duration_s?} }
```
