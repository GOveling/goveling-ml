#!/usr/bin/env python3
"""
🧮 City2Graph OR-Tools Service Integration
Wrapper seguro para ProfessionalItineraryOptimizer con circuit breaker y health checks

Basado en benchmarks que demuestran:
- 100% success rate OR-Tools vs 0% sistema clásico
- 4x más rápido (2000ms vs 8500ms)
- Distancias reales vs 0km del legacy system
- APIs funcionales vs múltiples errores legacy

Autor: Goveling ML Team - OR-Tools Integration
Fecha: Oct 19, 2025 - Post Benchmark Analysis
"""

import asyncio
import time
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass

from services.ortools_professional_optimizer import ProfessionalItineraryOptimizer

# OSRM imports for distance matrix
try:
    from services.osrm_service import OSRMService
    OSRM_AVAILABLE = True
except ImportError:
    OSRM_AVAILABLE = False
    logging.warning("⚠️ OSRM Service not available - will use euclidean distances")

logger = logging.getLogger(__name__)

@dataclass
class ORToolsMetrics:
    """Métricas de performance OR-Tools vs benchmarks"""
    execution_time_ms: float
    success: bool
    places_processed: int
    total_distance_km: float
    total_time_minutes: float
    benchmark_deviation: Dict[str, float]  # Diferencia vs benchmark expectations

class ORToolsCircuitBreaker:
    """
    🔄 Circuit Breaker específico para OR-Tools
    Más permisivo que legacy debido a demostrada confiabilidad
    """
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold  # Tolerante, OR-Tools demostró confiabilidad
        self.last_failure_time = None
        self.recovery_timeout = recovery_timeout    # Recovery rápido
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        # Métricas específicas OR-Tools
        self.success_count = 0
        self.total_executions = 0
        self.avg_execution_time = 2000.0  # Basado en benchmarks
        self.benchmark_expectations = {
            "avg_execution_time_ms": 2000,
            "success_rate": 1.0,
            "distance_calculation": True
        }
    
    async def execute_ortools(self, func, *args, **kwargs) -> Any:
        """
        🧮 Ejecutar función OR-Tools con circuit breaker
        """
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info("🔄 OR-Tools circuit breaker: HALF_OPEN - testing recovery")
                self.state = "HALF_OPEN"
            else:
                logger.warning("⛔ OR-Tools circuit breaker: OPEN - using fallback")
                raise CircuitBreakerOpenException("OR-Tools circuit breaker is OPEN")
        
        start_time = time.time()
        self.total_executions += 1
        
        try:
            result = await func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000
            
            # Success handling
            if self.state == "HALF_OPEN":
                logger.info("✅ OR-Tools recovery successful - circuit breaker CLOSED")
                self.state = "CLOSED"
                self.failure_count = 0
            
            self.success_count += 1
            self._update_avg_execution_time(execution_time)
            
            # Performance monitoring vs benchmarks
            await self._monitor_performance(execution_time, result)
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            logger.error(f"❌ OR-Tools error (#{self.failure_count}/{self.failure_threshold}): {str(e)}")
            
            if self.failure_count >= self.failure_threshold:
                logger.error("🚨 OR-Tools circuit breaker: OPENING due to failures")
                self.state = "OPEN"
            
            raise e
    
    def _update_avg_execution_time(self, execution_time: float):
        """Actualizar tiempo promedio de ejecución"""
        alpha = 0.1  # Peso para promedio móvil
        self.avg_execution_time = (1 - alpha) * self.avg_execution_time + alpha * execution_time
    
    async def _monitor_performance(self, execution_time: float, result: Dict):
        """Monitor performance vs benchmark expectations"""
        
        # Alert si diverge significativamente de benchmarks
        if execution_time > self.benchmark_expectations["avg_execution_time_ms"] * 2.5:
            logger.warning(f"🐌 OR-Tools slower than benchmark: {execution_time:.0f}ms vs {self.benchmark_expectations['avg_execution_time_ms']}ms expected")
        
        # Verificar que calculate distancias reales
        if result and isinstance(result, dict):
            total_distance = result.get("total_distance_km", 0)
            if total_distance == 0 and result.get("places_count", 0) > 3:
                logger.warning("⚠️ OR-Tools returned 0km distance - unexpected based on benchmarks")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Status de salud del circuit breaker"""
        success_rate = self.success_count / max(self.total_executions, 1)
        
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_rate": success_rate,
            "total_executions": self.total_executions,
            "avg_execution_time_ms": self.avg_execution_time,
            "vs_benchmark": {
                "execution_time_ratio": self.avg_execution_time / self.benchmark_expectations["avg_execution_time_ms"],
                "success_rate_vs_expected": success_rate / self.benchmark_expectations["success_rate"]
            },
            "healthy": self.state == "CLOSED" and success_rate > 0.9
        }

class CircuitBreakerOpenException(Exception):
    """Exception cuando circuit breaker está abierto"""
    pass

class City2GraphORToolsService:
    """
    🧮 Servicio principal de integración OR-Tools
    Wrapper seguro alrededor de ProfessionalItineraryOptimizer
    """
    
    def __init__(self):
        self.ortools_optimizer = None
        self.circuit_breaker = ORToolsCircuitBreaker()
        self.initialization_time = None
        self.health_check_cache = {"status": "unknown", "timestamp": 0}
        self.health_check_ttl = 300  # 5 minutes cache
        
        # Inicializar OSRM para matriz de distancias
        self.osrm_service = None
        if OSRM_AVAILABLE:
            try:
                self.osrm_service = OSRMService()
                logger.info("✅ OSRM Service initialized for distance matrix")
            except Exception as e:
                logger.warning(f"⚠️ OSRM initialization failed: {e} - using euclidean fallback")
        
        logger.info("🧮 Initializing City2Graph OR-Tools Service")
    
    async def initialize(self) -> bool:
        """Inicializar OR-Tools optimizer"""
        try:
            start_time = time.time()
            self.ortools_optimizer = ProfessionalItineraryOptimizer()
            self.initialization_time = time.time() - start_time
            
            logger.info(f"✅ OR-Tools optimizer initialized in {self.initialization_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OR-Tools optimizer: {e}")
            return False
    
    async def is_healthy(self) -> bool:
        """
        🏥 Health check OR-Tools
        Cache por 5 minutos para evitar overhead
        """
        current_time = time.time()
        
        # Return cached result if recent
        if current_time - self.health_check_cache["timestamp"] < self.health_check_ttl:
            return self.health_check_cache["status"] == "healthy"
        
        try:
            if not self.ortools_optimizer:
                await self.initialize()
                if not self.ortools_optimizer:
                    return False
            
            # Quick health check with minimal data
            test_places = [
                {"name": "Test A", "latitude": -33.4372, "longitude": -70.6506},
                {"name": "Test B", "latitude": -33.4263, "longitude": -70.6344}
            ]
            
            start_time = time.time()
            health_result = await asyncio.wait_for(
                self._quick_health_test(test_places),
                timeout=5.0  # Quick timeout for health check
            )
            health_time = (time.time() - start_time) * 1000
            
            is_healthy = (
                health_result is not None and 
                health_time < 5000 and  # Should be much faster than 5s
                self.circuit_breaker.state == "CLOSED"
            )
            
            # Cache result
            self.health_check_cache = {
                "status": "healthy" if is_healthy else "unhealthy",
                "timestamp": current_time,
                "health_time_ms": health_time
            }
            
            logger.info(f"🏥 OR-Tools health check: {'✅' if is_healthy else '❌'} ({health_time:.0f}ms)")
            return is_healthy
            
        except Exception as e:
            logger.warning(f"🏥 OR-Tools health check failed: {e}")
            self.health_check_cache = {
                "status": "unhealthy",
                "timestamp": current_time
            }
            return False
    
    async def _quick_health_test(self, test_places: List[Dict]) -> Optional[Dict]:
        """Test rápido de salud con datos mínimos"""
        try:
            if not self.ortools_optimizer:
                return None
                
            # Solo verificar que el optimizer se inicializó correctamente
            # Sin llamar a métodos complejos en health check
            if hasattr(self.ortools_optimizer, 'optimize_itinerary_advanced'):
                logger.info("✅ OR-Tools optimizer methods available")
                return {"status": "healthy", "methods_available": True}
            else:
                logger.warning("❌ OR-Tools optimizer missing expected methods")
                return None
            
        except Exception as e:
            logger.warning(f"🏥 Quick health test failed: {e}")
            return None
    
    async def get_distance_matrix(self, places: List[Dict]) -> Dict:
        """
        🗺️ Obtener matriz de distancias para OR-Tools
        Usa OSRM cuando está disponible, fallback a euclidiana
        """
        coordinates = [(place.get("lat", place.get("latitude", 0)), 
                       place.get("lon", place.get("longitude", 0))) for place in places]
        
        try:
            # Intentar usar OSRM para matriz real
            if self.osrm_service:
                logger.info(f"🗺️ Calculating distance matrix with OSRM for {len(places)} places")
                matrix_result = self.osrm_service.distance_matrix(coordinates)  # Método correcto
                
                if matrix_result and "distances" in matrix_result:
                    logger.info(f"✅ OSRM distance matrix: {len(matrix_result['distances'])}x{len(matrix_result['distances'][0])}")
                    return matrix_result
                else:
                    logger.warning("⚠️ OSRM returned empty matrix, using euclidean fallback")
            
            # Fallback a matriz euclidiana
            logger.info(f"📐 Using euclidean distance matrix for {len(places)} places")
            return self._create_euclidean_matrix(coordinates)
                
        except Exception as e:
            logger.warning(f"❌ Distance matrix calculation failed: {e}, using euclidean fallback")
            return self._create_euclidean_matrix(coordinates)
    
    def _create_euclidean_matrix(self, coordinates: List[Tuple[float, float]]) -> Dict:
        """Crear matriz de distancias euclidiana como fallback"""
        n = len(coordinates)
        distances = []
        durations = []
        
        for i in range(n):
            dist_row = []
            dur_row = []
            for j in range(n):
                if i == j:
                    dist_row.append(0.0)
                    dur_row.append(0.0)
                else:
                    # Calcular distancia haversine en km
                    dist = self._haversine_distance(coordinates[i], coordinates[j])
                    dist_row.append(dist)
                    # Estimar tiempo (asumiendo 50 km/h promedio)
                    dur_row.append(dist / 50.0 * 60.0)  # minutos
            distances.append(dist_row)
            durations.append(dur_row)
        
        return {
            "distances": distances,
            "durations": durations,
            "sources": list(range(n)),
            "destinations": list(range(n))
        }
    
    def _haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calcular distancia haversine entre dos coordenadas"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Radio de la Tierra en km
        R = 6371.0
        
        # Convertir grados a radianes
        lat1_r = math.radians(lat1)
        lon1_r = math.radians(lon1)
        lat2_r = math.radians(lat2)
        lon2_r = math.radians(lon2)
        
        # Diferencias
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        
        # Fórmula haversine
        a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    async def optimize_with_ortools(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        🧮 Optimización principal con OR-Tools
        Wrapper seguro con circuit breaker y métricas
        """
        if not await self.is_healthy():
            raise Exception("OR-Tools service is not healthy")
        
        start_time = time.time()
        
        try:
            # Ejecutar con circuit breaker
            result = await self.circuit_breaker.execute_ortools(
                self._execute_ortools_optimization,
                request
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Generar métricas de resultado
            metrics = self._generate_metrics(result, execution_time, request)
            
            # Agregar métricas al resultado
            result["ortools_metrics"] = metrics
            result["execution_meta"] = {
                "algorithm": "ortools_professional",
                "execution_time_ms": execution_time,
                "circuit_breaker_state": self.circuit_breaker.state,
                "vs_benchmark": metrics.benchmark_deviation
            }
            
            logger.info(f"🧮 OR-Tools optimization completed: "
                       f"{metrics.places_processed} places, "
                       f"{metrics.total_distance_km:.1f}km, "
                       f"{execution_time:.0f}ms")
            
            return result
            
        except CircuitBreakerOpenException as e:
            logger.error(f"⛔ OR-Tools circuit breaker open: {e}")
            raise
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"❌ OR-Tools optimization failed after {execution_time:.0f}ms: {e}")
            raise
    
    async def _execute_ortools_optimization(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecución real de optimización OR-Tools"""
        if not self.ortools_optimizer:
            await self.initialize()
            if not self.ortools_optimizer:
                raise Exception("Could not initialize OR-Tools optimizer")
        
        # Extraer datos del request para OR-Tools
        places = request.get("places", [])
        # NOTE (Fase 2.5 pendiente): accommodations con check_in/check_out se
        # extraen aquí pero NO se reenvían al optimizer todavía. La firma de
        # `optimize_itinerary_advanced` es single-day y no acepta este kwarg
        # (lanzaba TypeError). El anclaje por fecha en el path OR-Tools
        # requiere una refactor multi-día separada — ver Fase 2.5.
        accommodations = request.get("accommodations", [])
        preferences = request.get("preferences", {})

        # Obtener matriz de distancias
        logger.info(f"🗺️ Getting distance matrix for {len(places)} places")
        distance_matrix = await self.get_distance_matrix(places)

        # Ejecutar optimización con parámetros correctos
        logger.info(
            f"🧮 Executing OR-Tools optimization with pois={len(places)}, "
            f"accommodations_received={len(accommodations)} (date-anchoring TODO Fase 2.5)"
        )
        result = self.ortools_optimizer.optimize_itinerary_advanced(
            pois=places,  # Parámetro correcto: pois, no places
            distance_matrix=distance_matrix,  # Matriz de distancias requerida
            use_time_windows=True,
            start_time=f"{preferences.get('daily_start_hour', 9):02d}:00"
        )
        
        if not result:
            raise Exception("OR-Tools returned empty result")
        
        logger.info(f"✅ OR-Tools optimization completed successfully")
        return result
    
    def _generate_metrics(self, result: Dict, execution_time: float, request: Dict) -> ORToolsMetrics:
        """Generar métricas de performance vs benchmarks"""
        
        places_count = len(request.get("places", []))
        total_distance = result.get("total_distance_km", 0)
        total_time = result.get("total_time_minutes", 0)
        success = result.get("success", False)
        
        # Calcular desviación vs benchmarks
        benchmark_deviation = {
            "execution_time_ratio": execution_time / 2000.0,  # vs 2000ms benchmark
            "distance_realistic": total_distance > 0,  # vs 0km legacy system
            "success_vs_expected": 1.0 if success else 0.0  # vs 100% benchmark
        }
        
        return ORToolsMetrics(
            execution_time_ms=execution_time,
            success=success,
            places_processed=places_count,
            total_distance_km=total_distance,
            total_time_minutes=total_time,
            benchmark_deviation=benchmark_deviation
        )
    
    def get_service_status(self) -> Dict[str, Any]:
        """Status completo del servicio OR-Tools"""
        circuit_health = self.circuit_breaker.get_health_status()
        
        return {
            "service": "city2graph_ortools",
            "initialized": self.ortools_optimizer is not None,
            "initialization_time_s": self.initialization_time,
            "circuit_breaker": circuit_health,
            "health_cache": self.health_check_cache,
            "overall_healthy": (
                self.ortools_optimizer is not None and
                circuit_health["healthy"] and
                self.health_check_cache.get("status") == "healthy"
            ),
            "benchmark_compliance": {
                "avg_execution_time_ms": circuit_health["avg_execution_time_ms"],
                "vs_benchmark_2000ms": circuit_health["avg_execution_time_ms"] / 2000.0,
                "success_rate": circuit_health["success_rate"],
                "vs_benchmark_100pct": circuit_health["success_rate"] / 1.0
            }
        }

# Factory function para crear instancia singleton
_ortools_service_instance = None

async def get_ortools_service() -> City2GraphORToolsService:
    """Factory function para obtener instancia singleton del servicio"""
    global _ortools_service_instance
    
    if _ortools_service_instance is None:
        _ortools_service_instance = City2GraphORToolsService()
        await _ortools_service_instance.initialize()
    
    return _ortools_service_instance

# Health check endpoint helper
async def ortools_health_check() -> Dict[str, Any]:
    """Helper para health check endpoint"""
    try:
        service = await get_ortools_service()
        is_healthy = await service.is_healthy()
        status = service.get_service_status()
        
        return {
            "healthy": is_healthy,
            "status": status,
            "recommendation": (
                "OR-Tools ready for production use" if is_healthy 
                else "Use legacy fallback"
            )
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "recommendation": "Use legacy fallback"
        }