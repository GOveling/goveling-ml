#!/usr/bin/env python3
"""
🎯 OR-Tools Decision Algorithm
Algoritmo inteligente para decidir cuándo usar OR-Tools vs sistema legacy

Basado en benchmarks científicos que demuestran:
- OR-Tools: 100% success rate vs 0% sistema clásico
- OR-Tools: 4x más rápido (2000ms vs 8500ms)
- OR-Tools: Distancias reales vs 0km del legacy system
- OR-Tools: APIs funcionales vs múltiples errores legacy

Autor: Goveling ML Team - OR-Tools Integration
Fecha: Oct 19, 2025 - Post Benchmark Analysis
"""

import asyncio
import logging
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from math import sqrt

from settings import settings
from services.city2graph_ortools_service import get_ortools_service

logger = logging.getLogger(__name__)

@dataclass
class DecisionResult:
    """Resultado de decisión de algoritmo OR-Tools"""
    use_ortools: bool
    confidence_score: float  # 0.0 - 1.0
    reasons: List[str]
    complexity_score: float  # 0.0 - 10.0
    estimated_execution_time_ms: int
    expected_success_rate: float
    fallback_strategy: str
    decision_metadata: Dict[str, Any]

@dataclass
class ItineraryComplexity:
    """Análisis de complejidad de itinerario"""
    places_count: int
    days_count: int
    geographic_spread_km: float
    semantic_diversity: int
    time_constraints: int
    transport_complexity: str
    overall_score: float  # 0.0 - 10.0

class ORToolsDecisionEngine:
    """
    🧠 Motor de decisión inteligente para OR-Tools
    Determina cuándo usar OR-Tools vs sistema legacy basado en benchmarks y análisis
    """
    
    def __init__(self):
        self.decision_cache = {}
        self.cache_ttl = 600  # 10 minutos
        self.performance_history = []
        self.last_health_check = 0
        self.ortools_health_status = None
        
        logger.info("🧠 OR-Tools Decision Engine initialized")
    
    async def should_use_ortools(self, request_data: Dict[str, Any]) -> DecisionResult:
        """
        🎯 Decisión principal: ¿Usar OR-Tools o sistema legacy?
        
        Args:
            request_data: Datos del request de itinerario
            
        Returns:
            DecisionResult con decisión y metadata
        """
        start_time = time.time()
        
        try:
            # Generar cache key para decisión
            cache_key = self._generate_cache_key(request_data)
            
            # Check cache primero
            if cache_key in self.decision_cache:
                cached_result = self.decision_cache[cache_key]
                if time.time() - cached_result["timestamp"] < self.cache_ttl:
                    logger.info(f"🎯 Decision cache hit: {'OR-Tools' if cached_result['result'].use_ortools else 'Legacy'}")
                    return cached_result["result"]
            
            # Análisis de complejidad
            complexity = await self._analyze_complexity(request_data)
            
            # Check de salud OR-Tools
            ortools_healthy = await self._check_ortools_health()
            
            # Feature flags y configuración
            feature_flags = self._check_feature_flags(request_data)
            
            # Análisis geográfico
            geo_analysis = self._analyze_geography(request_data)
            
            # Decisión basada en todos los factores
            decision = await self._make_decision(
                complexity=complexity,
                ortools_healthy=ortools_healthy,
                feature_flags=feature_flags,
                geo_analysis=geo_analysis,
                request_data=request_data
            )
            
            # Cache resultado
            self.decision_cache[cache_key] = {
                "result": decision,
                "timestamp": time.time()
            }
            
            decision_time = (time.time() - start_time) * 1000
            logger.info(f"🎯 Decision made in {decision_time:.0f}ms: "
                       f"{'✅ OR-Tools' if decision.use_ortools else '🔧 Legacy'} "
                       f"(confidence: {decision.confidence_score:.2f})")
            
            # Track decisión para análisis
            if settings.ORTOOLS_TRACK_PERFORMANCE:
                await self._track_decision(decision, complexity, decision_time)
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ Decision engine error: {e}")
            # Fallback conservador a legacy en caso de error
            return DecisionResult(
                use_ortools=False,
                confidence_score=0.0,
                reasons=[f"decision_engine_error: {str(e)}"],
                complexity_score=0.0,
                estimated_execution_time_ms=8500,  # Basado en benchmark legacy
                expected_success_rate=0.0,  # Basado en benchmark legacy
                fallback_strategy="legacy_safe_fallback",
                decision_metadata={"error": str(e)}
            )
    
    async def _analyze_complexity(self, request_data: Dict[str, Any]) -> ItineraryComplexity:
        """📊 Análisis de complejidad del itinerario"""
        
        places = request_data.get("places", [])
        start_date = request_data.get("start_date")
        end_date = request_data.get("end_date")
        
        places_count = len(places)
        
        # Calcular días
        try:
            if isinstance(start_date, str):
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            else:
                start_dt = start_date
            
            if isinstance(end_date, str):
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            else:
                end_dt = end_date
                
            days_count = (end_dt - start_dt).days + 1
        except:
            days_count = 1
        
        # Análisis geográfico
        geographic_spread = self._calculate_geographic_spread(places)
        
        # Diversidad semántica
        semantic_diversity = self._calculate_semantic_diversity(places)
        
        # Restricciones temporales
        time_constraints = self._count_time_constraints(request_data)
        
        # Complejidad de transporte
        transport_complexity = self._analyze_transport_complexity(request_data)
        
        # Score general (0-10)
        overall_score = self._calculate_overall_complexity(
            places_count, days_count, geographic_spread, 
            semantic_diversity, time_constraints, transport_complexity
        )
        
        return ItineraryComplexity(
            places_count=places_count,
            days_count=days_count,
            geographic_spread_km=geographic_spread,
            semantic_diversity=semantic_diversity,
            time_constraints=time_constraints,
            transport_complexity=transport_complexity,
            overall_score=overall_score
        )
    
    def _calculate_geographic_spread(self, places: List[Dict]) -> float:
        """Calcular dispersión geográfica en km"""
        if len(places) < 2:
            return 0.0
        
        # Encontrar bounding box
        lats = [p.get("lat", p.get("latitude", 0)) for p in places if p.get("lat") or p.get("latitude")]
        lons = [p.get("lon", p.get("longitude", 0)) for p in places if p.get("lon") or p.get("longitude")]
        
        if not lats or not lons:
            return 0.0
        
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        
        # Aproximación rough de km (111km por grado lat, varía por lon)
        lat_km = lat_range * 111
        lon_km = lon_range * 111 * 0.8  # Factor aprox para latitudes de Chile
        
        return sqrt(lat_km**2 + lon_km**2)
    
    def _calculate_semantic_diversity(self, places: List[Dict]) -> int:
        """Contar tipos semánticos diferentes"""
        def get_place_attr(place, *attrs):
            """Helper para obtener atributos de Place objects o dicts"""
            for attr in attrs:
                if hasattr(place, attr):
                    value = getattr(place, attr)
                    if value is not None:
                        return value
                elif isinstance(place, dict) and attr in place:
                    return place[attr]
            return "unknown"
        
        types = set()
        for place in places:
            place_type = get_place_attr(place, "type", "place_type")
            types.add(place_type)
        return len(types)
    
    def _count_time_constraints(self, request_data: Dict) -> int:
        """Contar restricciones temporales"""
        constraints = 0
        
        if request_data.get("daily_start_hour"):
            constraints += 1
        if request_data.get("daily_end_hour"):
            constraints += 1
        if request_data.get("time_windows"):
            constraints += len(request_data["time_windows"])
        if request_data.get("fixed_activities"):
            constraints += len(request_data["fixed_activities"])
            
        return constraints
    
    def _analyze_transport_complexity(self, request_data: Dict) -> str:
        """Analizar complejidad de transporte"""
        transport_mode = request_data.get("transport_mode", "driving")
        
        if transport_mode == "walking":
            return "simple"
        elif transport_mode in ["driving", "bicycling"]:
            return "medium"
        elif transport_mode in ["transit", "mixed"]:
            return "complex"
        else:
            return "medium"
    
    def _calculate_overall_complexity(self, places_count: int, days_count: int, 
                                    geographic_spread: float, semantic_diversity: int,
                                    time_constraints: int, transport_complexity: str) -> float:
        """Calcular score de complejidad general (0-10)"""
        
        score = 0.0
        
        # Peso por número de lugares (factor más importante)
        if places_count >= 10:
            score += 3.0
        elif places_count >= 6:
            score += 2.0
        elif places_count >= 3:
            score += 1.0
        
        # Peso por días
        if days_count >= 5:
            score += 2.0
        elif days_count >= 3:
            score += 1.0
        elif days_count >= 2:
            score += 0.5
        
        # Peso por dispersión geográfica
        if geographic_spread >= 100:
            score += 2.0
        elif geographic_spread >= 50:
            score += 1.5
        elif geographic_spread >= 20:
            score += 1.0
        elif geographic_spread >= 5:
            score += 0.5
        
        # Diversidad semántica
        score += min(semantic_diversity * 0.3, 1.5)
        
        # Restricciones temporales
        score += min(time_constraints * 0.2, 1.0)
        
        # Complejidad de transporte
        transport_weights = {"simple": 0, "medium": 0.5, "complex": 1.0}
        score += transport_weights.get(transport_complexity, 0.5)
        
        return min(score, 10.0)
    
    async def _check_ortools_health(self) -> bool:
        """🏥 Check de salud OR-Tools con cache"""
        current_time = time.time()
        
        # Cache health check para evitar overhead
        if (current_time - self.last_health_check) < 60:  # Cache 1 minuto
            return self.ortools_health_status or False
        
        try:
            ortools_service = await get_ortools_service()
            is_healthy = await ortools_service.is_healthy()
            
            self.ortools_health_status = is_healthy
            self.last_health_check = current_time
            
            logger.info(f"🏥 OR-Tools health: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")
            return is_healthy
            
        except Exception as e:
            logger.warning(f"🏥 OR-Tools health check failed: {e}")
            self.ortools_health_status = False
            self.last_health_check = current_time
            return False
    
    def _check_feature_flags(self, request_data: Dict) -> Dict[str, Any]:
        """🚩 Check de feature flags y configuración"""
        
        # Master switch
        if not settings.ENABLE_ORTOOLS:
            return {
                "enabled": False,
                "reason": "ENABLE_ORTOOLS=False",
                "user_eligible": False,
                "geo_eligible": False
            }
        
        # User percentage rollout
        user_eligible = self._check_user_eligibility(request_data)
        
        # Geografía elegible
        geo_eligible = self._check_geo_eligibility(request_data)
        
        return {
            "enabled": settings.ENABLE_ORTOOLS,
            "reason": "feature_flags_passed",
            "user_eligible": user_eligible,
            "geo_eligible": geo_eligible
        }
    
    def _check_user_eligibility(self, request_data: Dict) -> bool:
        """👤 Check si usuario es elegible para OR-Tools"""
        if settings.ORTOOLS_USER_PERCENTAGE >= 100:
            return True
        
        if settings.ORTOOLS_USER_PERCENTAGE <= 0:
            return False
        
        # Hash user identifier para distribución consistente
        user_id = request_data.get("user_id", request_data.get("session_id", "anonymous"))
        user_hash = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        
        return (user_hash % 100) < settings.ORTOOLS_USER_PERCENTAGE
    
    def _check_geo_eligibility(self, request_data: Dict) -> bool:
        """🌍 Check si geografía es elegible para OR-Tools"""
        
        # Si no hay restricciones geográficas, todo elegible
        if not settings.ORTOOLS_CITIES:
            return True
        
        # Detectar ciudad del request
        detected_city = self._detect_city(request_data)
        
        # Check ciudades permitidas
        allowed_cities = [city.strip().lower() for city in settings.ORTOOLS_CITIES.split(",") if city.strip()]
        excluded_cities = [city.strip().lower() for city in settings.ORTOOLS_EXCLUDE_CITIES.split(",") if city.strip()]
        
        if detected_city:
            city_lower = detected_city.lower()

            # Synthetic cluster IDs produced by _detect_city when no named city
            # is available (e.g. "multi_city_2_clusters", "city_cluster_-33.44_-70.65",
            # "location_...", "single_location").  These are not real city names so
            # they will never appear in allowed_cities — skip the allowlist check and
            # let OR-Tools run.  Exclusions still apply if the raw name matches.
            _SYNTHETIC_PREFIXES = ("multi_city_", "city_cluster_", "location_", "single_location")
            is_synthetic = any(city_lower.startswith(p) for p in _SYNTHETIC_PREFIXES)

            # Check exclusiones primero (applies even for synthetic IDs)
            if city_lower in excluded_cities:
                return False

            # Check inclusiones — skip for synthetic IDs (can't name-match a cluster)
            if allowed_cities and not is_synthetic and city_lower not in allowed_cities:
                return False

        return True
    
    def _detect_city(self, request_data: Dict) -> Optional[str]:
        """🏙️ Detectar ciudad principal del itinerario usando clustering automático"""
        places = request_data.get("places", [])
        
        if not places:
            return None
        
        # Buscar ciudad en metadatos de lugares
        def get_place_attr(place, *attrs):
            """Helper para obtener atributos de Place objects o dicts"""
            for attr in attrs:
                if hasattr(place, attr):
                    value = getattr(place, attr)
                    if value is not None:
                        return value
                elif isinstance(place, dict) and attr in place:
                    return place[attr]
            return None
        
        for place in places:
            city = get_place_attr(place, "city", "locality")
            if city:
                return city
        
        # 🔧 NUEVO: Detección automática usando clustering geográfico
        try:
            clusters = self._detect_geographic_clusters(places)
            
            if len(clusters) == 1:
                # Una sola ciudad detectada
                cluster = clusters[0]
                return f"city_cluster_{cluster['center_lat']:.2f}_{cluster['center_lon']:.2f}"
            elif len(clusters) > 1:
                # Múltiples ciudades - retornar la más grande
                largest_cluster = max(clusters, key=lambda c: c['places_count'])
                return f"multi_city_{len(clusters)}_clusters"
            else:
                # No se pudieron formar clusters
                return "single_location"
                
        except Exception as e:
            logger.warning(f"⚠️ Error en clustering automático: {e}")
        
        # Fallback: usar coordenadas promedio
        lats = [p.get("lat", p.get("latitude", 0)) for p in places if p.get("lat") or p.get("latitude")]
        lons = [p.get("lon", p.get("longitude", 0)) for p in places if p.get("lon") or p.get("longitude")]
        
        if lats and lons:
            avg_lat = sum(lats) / len(lats)
            avg_lon = sum(lons) / len(lons)
            return f"location_{avg_lat:.2f}_{avg_lon:.2f}"
        
        return None
    
    def _detect_geographic_clusters(self, places: List[Dict]) -> List[Dict]:
        """🗺️ Detectar clusters geográficos automáticamente usando DBSCAN"""
        from sklearn.cluster import DBSCAN
        import numpy as np
        from math import radians
        
        if len(places) < 2:
            return []
        
        # Extraer coordenadas válidas
        coordinates = []
        valid_places = []
        
        def get_place_attr(place, *attrs):
            """Helper para obtener atributos de Place objects o dicts"""
            for attr in attrs:
                if hasattr(place, attr):
                    value = getattr(place, attr)
                    if value is not None:
                        return value
                elif isinstance(place, dict) and attr in place:
                    return place[attr]
            return None
        
        for place in places:
            lat = get_place_attr(place, "lat", "latitude")
            lon = get_place_attr(place, "lon", "longitude")
            
            if lat is not None and lon is not None:
                try:
                    lat, lon = float(lat), float(lon)
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        coordinates.append([radians(lat), radians(lon)])
                        valid_places.append(place)
                except (ValueError, TypeError):
                    continue
        
        if len(coordinates) < 2:
            return []
        
        # DBSCAN con métrica haversine (distancia en esfera terrestre)
        # eps = 100km en radianes (100km / 6371km radio tierra)
        eps_km = 100  # 100km threshold para considerar mismo cluster
        eps_radians = eps_km / 6371.0
        
        clustering = DBSCAN(
            eps=eps_radians,
            min_samples=1,  # Mínimo 1 lugar por cluster
            metric='haversine'
        ).fit(coordinates)
        
        # Agrupar por clusters
        clusters_dict = {}
        for i, label in enumerate(clustering.labels_):
            if label == -1:  # Noise points - crear cluster individual
                label = f"noise_{i}"
            
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append((valid_places[i], coordinates[i]))
        
        # Convertir a formato de retorno
        clusters = []
        for cluster_id, cluster_places in clusters_dict.items():
            if len(cluster_places) == 0:
                continue
                
            # Calcular centro del cluster
            lats = [place[1][0] for place in cluster_places]  # radianes
            lons = [place[1][1] for place in cluster_places]  # radianes
            
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
            
            # Convertir de radianes a grados
            center_lat_deg = center_lat * 180 / 3.14159
            center_lon_deg = center_lon * 180 / 3.14159
            
            clusters.append({
                'cluster_id': cluster_id,
                'places_count': len(cluster_places),
                'center_lat': center_lat_deg,
                'center_lon': center_lon_deg,
                'places': [place[0] for place in cluster_places]
            })
        
        return clusters
    
    def _analyze_geography(self, request_data: Dict) -> Dict[str, Any]:
        """🗺️ Análisis geográfico del itinerario"""
        places = request_data.get("places", [])
        
        if not places:
            return {"valid": False, "reason": "no_places"}
        
        geographic_spread = self._calculate_geographic_spread(places)
        detected_city = self._detect_city(request_data)
        
        # Determinar si es apropiado para OR-Tools
        appropriate = True
        reasons = []
        
        if geographic_spread > settings.ORTOOLS_MAX_DISTANCE_KM:
            appropriate = False
            reasons.append(f"spread_too_large_{geographic_spread:.0f}km")
        
        if len(places) > settings.ORTOOLS_MAX_PLACES:
            appropriate = False
            reasons.append(f"too_many_places_{len(places)}")
        
        return {
            "valid": True,
            "appropriate_for_ortools": appropriate,
            "reasons": reasons,
            "geographic_spread_km": geographic_spread,
            "detected_city": detected_city,
            "places_count": len(places)
        }
    
    async def _make_decision(self, complexity: ItineraryComplexity, ortools_healthy: bool,
                           feature_flags: Dict, geo_analysis: Dict, request_data: Dict) -> DecisionResult:
        """⚖️ Decisión final basada en todos los factores"""
        
        use_ortools = False
        confidence_score = 0.0
        reasons = []
        
        # Factor 1: Feature flags (requerimiento básico)
        if not feature_flags["enabled"]:
            reasons.append(f"feature_disabled: {feature_flags['reason']}")
            use_ortools = False
            confidence_score = 0.0
        elif not feature_flags["user_eligible"]:
            reasons.append("user_not_in_rollout_percentage")
            use_ortools = False
            confidence_score = 0.0
        elif not feature_flags["geo_eligible"]:
            reasons.append("geography_not_eligible")
            use_ortools = False
            confidence_score = 0.0
        # Factor 2: Salud OR-Tools
        elif not ortools_healthy:
            reasons.append("ortools_service_unhealthy")
            use_ortools = False
            confidence_score = 0.0
        # Factor 3: Geografía apropiada
        elif not geo_analysis.get("appropriate_for_ortools", True):
            reasons.extend(geo_analysis.get("reasons", []))
            use_ortools = False
            confidence_score = 0.0
        # Factor 4: Complejidad (criterio principal basado en benchmarks)
        else:
            # Basado en benchmarks exitosos:
            # - Valparaíso: 6 lugares = éxito
            # - Santiago: 8 lugares = éxito
            # - Multi-city: 6 lugares = éxito (aunque 3 dropped)
            
            confidence_score = 0.5  # Base confidence
            
            # Lugares suficientes (factor más importante)
            if complexity.places_count >= settings.ORTOOLS_MIN_PLACES:
                reasons.append(f"sufficient_places_{complexity.places_count}")
                confidence_score += 0.3
            elif complexity.places_count >= 3:  # Mínimo razonable
                reasons.append(f"moderate_places_{complexity.places_count}")
                confidence_score += 0.1
            else:
                reasons.append(f"too_few_places_{complexity.places_count}")
                confidence_score = 0.0
            
            # Días (OR-Tools maneja bien casos simples)
            if complexity.days_count >= settings.ORTOOLS_MIN_DAYS:
                reasons.append(f"multi_day_{complexity.days_count}")
                confidence_score += 0.1
            else:
                reasons.append(f"single_day_ok")
                confidence_score += 0.05
            
            # Complejidad general (OR-Tools excels en casos complejos)
            if complexity.overall_score >= 5.0:
                reasons.append(f"high_complexity_{complexity.overall_score:.1f}")
                confidence_score += 0.1
            elif complexity.overall_score >= 3.0:
                reasons.append(f"medium_complexity_{complexity.overall_score:.1f}")
                confidence_score += 0.05
            
            # Decisión final
            if confidence_score >= 0.7:
                use_ortools = True
                reasons.append("high_confidence_ortools")
            elif confidence_score >= 0.5:
                use_ortools = True
                reasons.append("medium_confidence_ortools")
            else:
                use_ortools = False
                reasons.append("low_confidence_fallback_legacy")
                confidence_score = max(0.1, confidence_score)  # Mínimo confidence para legacy
        
        # Estimaciones basadas en benchmarks
        if use_ortools:
            estimated_time = settings.ORTOOLS_EXPECTED_EXEC_TIME_MS
            expected_success = 1.0  # Basado en benchmark 100% success
            fallback_strategy = "ortools_with_legacy_fallback"
        else:
            estimated_time = 8500  # Basado en benchmark legacy
            expected_success = 0.1  # Conservador, benchmark mostró 0% pero puede variar
            fallback_strategy = "legacy_only"
        
        # Metadata adicional
        decision_metadata = {
            "complexity_analysis": complexity,
            "ortools_healthy": ortools_healthy,
            "feature_flags": feature_flags,
            "geo_analysis": geo_analysis,
            "algorithm_version": "1.0_post_benchmark",
            "benchmark_basis": "2025_10_19_ortools_vs_classic"
        }
        
        return DecisionResult(
            use_ortools=use_ortools,
            confidence_score=confidence_score,
            reasons=reasons,
            complexity_score=complexity.overall_score,
            estimated_execution_time_ms=estimated_time,
            expected_success_rate=expected_success,
            fallback_strategy=fallback_strategy,
            decision_metadata=decision_metadata
        )
    
    def _generate_cache_key(self, request_data: Dict) -> str:
        """Generar cache key para decisión"""
        # Key basado en factores relevantes para decisión
        factors = {
            "places_count": len(request_data.get("places", [])),
            "start_date": str(request_data.get("start_date", "")),
            "end_date": str(request_data.get("end_date", "")),
            "transport_mode": request_data.get("transport_mode", ""),
            "user_id": request_data.get("user_id", "anonymous"),
            "enable_ortools": settings.ENABLE_ORTOOLS,
            "user_percentage": settings.ORTOOLS_USER_PERCENTAGE
        }
        
        factors_str = str(sorted(factors.items()))
        return hashlib.md5(factors_str.encode()).hexdigest()[:16]
    
    async def _track_decision(self, decision: DecisionResult, complexity: ItineraryComplexity, decision_time_ms: float):
        """📊 Track decisión para análisis futuro"""
        
        tracking_data = {
            "timestamp": datetime.now().isoformat(),
            "decision": decision.use_ortools,
            "confidence": decision.confidence_score,
            "complexity_score": complexity.overall_score,
            "places_count": complexity.places_count,
            "days_count": complexity.days_count,
            "decision_time_ms": decision_time_ms,
            "reasons": decision.reasons,
            "estimated_exec_time": decision.estimated_execution_time_ms,
            "expected_success_rate": decision.expected_success_rate
        }
        
        # Log para análisis (puede expandirse a base de datos)
        logger.info(f"📊 Decision tracked: {tracking_data}")
        
        # Mantener history limitado en memoria
        self.performance_history.append(tracking_data)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]  # Keep last 500
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """📈 Estadísticas de decisiones tomadas"""
        if not self.performance_history:
            return {"message": "No decisions tracked yet"}
        
        recent_decisions = self.performance_history[-100:]  # Last 100
        
        ortools_decisions = [d for d in recent_decisions if d["decision"]]
        legacy_decisions = [d for d in recent_decisions if not d["decision"]]
        
        return {
            "total_decisions": len(recent_decisions),
            "ortools_percentage": len(ortools_decisions) / len(recent_decisions) * 100,
            "legacy_percentage": len(legacy_decisions) / len(recent_decisions) * 100,
            "avg_confidence_ortools": sum(d["confidence"] for d in ortools_decisions) / max(len(ortools_decisions), 1),
            "avg_complexity_ortools": sum(d["complexity_score"] for d in ortools_decisions) / max(len(ortools_decisions), 1),
            "avg_decision_time_ms": sum(d["decision_time_ms"] for d in recent_decisions) / len(recent_decisions),
            "cache_hit_rate": len(self.decision_cache) / max(len(recent_decisions), 1) * 100
        }

# Factory function para instancia singleton
_decision_engine_instance = None

async def get_decision_engine() -> ORToolsDecisionEngine:
    """Factory function para obtener instancia singleton"""
    global _decision_engine_instance
    
    if _decision_engine_instance is None:
        _decision_engine_instance = ORToolsDecisionEngine()
    
    return _decision_engine_instance

# Helper function para uso directo
async def should_use_ortools(request_data: Dict[str, Any]) -> DecisionResult:
    """
    🎯 Helper function para decisión OR-Tools
    
    Args:
        request_data: Datos del request de itinerario
        
    Returns:
        DecisionResult con decisión y metadata
    """
    engine = await get_decision_engine()
    return await engine.should_use_ortools(request_data)