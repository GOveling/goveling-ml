#!/usr/bin/env python3
"""
🧮 OR-Tools Distance Matrix Cache Service - Week 4 Performance Optimization
Optimiza performance de OR-Tools mediante cache inteligente de matrices de distancia
"""

import logging
import hashlib
import json
import asyncio
import aiohttp
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
from geopy.distance import geodesic
import time

from settings import settings

# TODO producción: migrar a instancia OSRM propia cuando superes 1000 requests/día.
# El servidor público demo no tiene SLA y puede bloquear IPs con tráfico sostenido.
# Ver: https://github.com/Project-OSRM/osrm-backend
_OSRM_PUBLIC_URL = "https://router.project-osrm.org"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CachedDistanceMatrix:
    """Matriz de distancia cached con metadata"""
    matrix_data: List[List[float]]
    places_hash: str
    timestamp: datetime
    source: str  # 'ortools', 'osrm', 'geodesic'
    cache_ttl: int
    metadata: Dict[str, Any]
    
    @property
    def is_expired(self) -> bool:
        """Verificar si el cache ha expirado"""
        return datetime.now() > self.timestamp + timedelta(seconds=self.cache_ttl)
    
    @property
    def age_seconds(self) -> int:
        """Edad del cache en segundos"""
        return int((datetime.now() - self.timestamp).total_seconds())

class ORToolsDistanceCache:
    """
    🚀 Cache optimizado de matrices de distancia para OR-Tools
    
    Features Week 4:
    - Cache inteligente con TTL configurable
    - Múltiples fuentes (OSRM, geodesic, OR-Tools)
    - Invalidación automática por cambios geográficos
    - Estadísticas de performance
    - Paralelización de cálculos
    """
    
    def __init__(self):
        self.cache: Dict[str, CachedDistanceMatrix] = {}
        self.max_cache_size = 1000
        self.cache_ttl = settings.ORTOOLS_DISTANCE_CACHE_TTL
        self.stats = {
            "hits": 0,
            "misses": 0,
            "cache_size": 0,
            "avg_calculation_time_ms": 0.0,
            "total_matrices_cached": 0
        }
        
        # Services para cálculo de distancias
        self._init_distance_services()
        
        logger.info(f"🧮 ORToolsDistanceCache initialized - TTL: {self.cache_ttl}s, Max size: {self.max_cache_size}")
    
    def _init_distance_services(self):
        """Inicializar servicios de cálculo de distancia"""
        try:
            # Importar servicios solo cuando están disponibles
            from services.city2graph_real_complete import RealCity2GraphService
            self.osrm_service = RealCity2GraphService()
            self.has_osrm = True
            logger.info("✅ OSRM service available for distance cache")
        except Exception as e:
            self.has_osrm = False
            logger.warning(f"⚠️ OSRM service not available: {e}")
    
    def _generate_places_hash(self, places: List[Dict]) -> str:
        """
        Generar hash único para conjunto de lugares
        Considera lat, lon y orden para cache key
        """
        # Normalizar places para hash consistente
        normalized_places = []
        for place in places:
            normalized_places.append({
                'lat': round(place['lat'], 6),
                'lon': round(place['lon'], 6),
                'name': place.get('name', '')
            })
        
        # Ordenar por coordenadas para hash consistente
        normalized_places.sort(key=lambda p: (p['lat'], p['lon']))
        
        places_str = json.dumps(normalized_places, sort_keys=True)
        return hashlib.md5(places_str.encode()).hexdigest()
    
    async def get_distance_matrix(self, places: List[Dict], 
                                source_preference: str = "auto") -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Obtener matriz de distancia con cache inteligente
        
        Args:
            places: Lista de lugares con lat, lon, name
            source_preference: 'auto', 'osrm', 'geodesic', 'cached_only'
            
        Returns:
            (matrix, metadata) - Matriz de distancias y metadata del cache
        """
        start_time = time.time()
        
        # Generar cache key
        places_hash = self._generate_places_hash(places)
        
        # Intentar obtener desde cache
        cached_result = await self._get_from_cache(places_hash)
        if cached_result:
            self.stats["hits"] += 1
            
            calculation_time = (time.time() - start_time) * 1000
            metadata = {
                "source": "cache",
                "cache_age_seconds": cached_result.age_seconds,
                "calculation_time_ms": calculation_time,
                "original_source": cached_result.source,
                "cache_hit": True
            }
            
            logger.debug(f"🎯 Cache HIT - {len(places)} places, age: {cached_result.age_seconds}s")
            return cached_result.matrix_data, metadata
        
        # Cache miss - calcular nueva matriz
        self.stats["misses"] += 1
        logger.debug(f"📊 Cache MISS - calculating new matrix for {len(places)} places")
        
        # Calcular matriz según preferencia
        matrix, source_used = await self._calculate_distance_matrix(places, source_preference)
        
        # Guardar en cache
        await self._save_to_cache(places_hash, matrix, source_used, places)
        
        calculation_time = (time.time() - start_time) * 1000
        self.stats["avg_calculation_time_ms"] = (
            (self.stats["avg_calculation_time_ms"] + calculation_time) / 2
        )
        
        metadata = {
            "source": source_used,
            "calculation_time_ms": calculation_time,
            "cache_hit": False,
            "places_count": len(places)
        }
        
        logger.info(f"🧮 New matrix calculated - {len(places)} places, {calculation_time:.1f}ms, source: {source_used}")
        return matrix, metadata
    
    async def _get_from_cache(self, places_hash: str) -> Optional[CachedDistanceMatrix]:
        """Obtener matriz desde cache si existe y no ha expirado"""
        if places_hash not in self.cache:
            return None
        
        cached_matrix = self.cache[places_hash]
        
        # Verificar expiración
        if cached_matrix.is_expired:
            logger.debug(f"🗑️ Removing expired cache entry - age: {cached_matrix.age_seconds}s")
            del self.cache[places_hash]
            self.stats["cache_size"] = len(self.cache)
            return None
        
        return cached_matrix
    
    async def _calculate_distance_matrix(self, places: List[Dict], 
                                       source_preference: str) -> Tuple[List[List[float]], str]:
        """
        Calcular matriz de distancia usando la mejor fuente disponible
        """
        places_count = len(places)
        
        # Estrategia de fuente según disponibilidad y preferencia
        if source_preference == "auto":
            # Para pocos lugares, geodesic es suficiente y rápido
            if places_count <= 5:
                source_preference = "geodesic"
            # Para más lugares, preferir OSRM si está disponible
            elif self.has_osrm:
                source_preference = "osrm"
            else:
                source_preference = "geodesic"
        
        # Intentar fuente preferida
        try:
            if source_preference == "osrm" and self.has_osrm:
                return await self._calculate_osrm_matrix(places), "osrm"
            elif source_preference == "geodesic":
                return await self._calculate_geodesic_matrix(places), "geodesic"
            else:
                # Fallback a geodesic
                logger.warning(f"Source {source_preference} not available, using geodesic fallback")
                return await self._calculate_geodesic_matrix(places), "geodesic"
                
        except Exception as e:
            logger.error(f"❌ Error calculating matrix with {source_preference}: {e}")
            # Fallback a geodesic como último recurso
            if source_preference != "geodesic":
                logger.info("🔄 Falling back to geodesic calculation")
                return await self._calculate_geodesic_matrix(places), "geodesic_fallback"
            else:
                raise e
    
    async def _calculate_osrm_table(self, places: List[Dict]) -> List[List[float]]:
        """
        Calcular matriz NxN completa con una sola llamada a OSRM /table.
        Para N lugares: 1 request vs N*(N-1)/2 pairwise calls.
        """
        n = len(places)
        # OSRM espera coordenadas en formato lon,lat (no lat,lon)
        coords_str = ";".join(f"{p['lon']},{p['lat']}" for p in places)
        url = f"{_OSRM_PUBLIC_URL}/table/v1/driving/{coords_str}"

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params={"annotations": "duration,distance"},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                if response.status != 200:
                    raise Exception(f"OSRM /table HTTP {response.status}")

                data = await response.json()

                if data.get("code") != "Ok":
                    raise Exception(
                        f"OSRM /table code={data.get('code')}: {data.get('message', '')}"
                    )

                # Preferir distances (metros → km); si el servidor no las devuelve
                # usar durations (segundos) convertidos con velocidad media urbana.
                if data.get("distances"):
                    logger.info(f"OSRM /table: {n} lugares en 1 request (distances)")
                    return [[d / 1000.0 for d in row] for row in data["distances"]]
                elif data.get("durations"):
                    logger.info(f"OSRM /table: {n} lugares en 1 request (durations→km)")
                    avg_speed_kmh = 40.0
                    return [
                        [(s / 3600.0) * avg_speed_kmh for s in row]
                        for row in data["durations"]
                    ]
                else:
                    raise Exception("OSRM /table: respuesta sin distances ni durations")

    async def _calculate_osrm_matrix(self, places: List[Dict]) -> List[List[float]]:
        """
        Calcular matriz usando OSRM.
        Intenta /table (1 request) y cae silenciosamente a pairwise si falla.
        """
        if not self.has_osrm:
            raise Exception("OSRM service not available")

        try:
            return await self._calculate_osrm_table(places)
        except Exception as e:
            logger.warning(f"OSRM /table falló, usando pairwise: {e}")

        # Fallback: pairwise calls (comportamiento original)
        try:
            if len(places) > 10 and settings.ORTOOLS_ENABLE_PARALLEL_OPTIMIZATION:
                return await self._calculate_osrm_parallel(places)
            else:
                return await self._calculate_osrm_sequential(places)
        except Exception as e:
            logger.error(f"❌ OSRM pairwise también falló: {e}")
            raise
    
    async def _calculate_osrm_sequential(self, places: List[Dict]) -> List[List[float]]:
        """Cálculo secuencial OSRM"""
        matrix = []
        
        for i, origin in enumerate(places):
            row = []
            for j, destination in enumerate(places):
                if i == j:
                    row.append(0.0)
                else:
                    try:
                        # Usar OSRM para distancia real
                        distance_km = await self._get_osrm_distance(origin, destination)
                        row.append(distance_km)
                    except Exception as e:
                        logger.warning(f"OSRM failed for {i}→{j}: {e}, using geodesic")
                        # Fallback a geodesic para este par
                        geodesic_dist = geodesic(
                            (origin['lat'], origin['lon']),
                            (destination['lat'], destination['lon'])
                        ).kilometers
                        row.append(geodesic_dist)
            
            matrix.append(row)
        
        return matrix
    
    async def _calculate_osrm_parallel(self, places: List[Dict]) -> List[List[float]]:
        """Cálculo paralelo OSRM para mejor performance"""
        places_count = len(places)
        matrix = [[0.0] * places_count for _ in range(places_count)]
        
        # Crear tareas para cálculos paralelos
        tasks = []
        for i in range(places_count):
            for j in range(i + 1, places_count):  # Solo calcular triángulo superior
                task = asyncio.create_task(
                    self._get_osrm_distance_with_indices(places[i], places[j], i, j)
                )
                tasks.append(task)
        
        # Ejecutar en paralelo con límite de concurrencia
        semaphore = asyncio.Semaphore(settings.ORTOOLS_MAX_PARALLEL_REQUESTS)
        
        async def limited_task(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[limited_task(task) for task in tasks], return_exceptions=True)
        
        # Llenar matriz con resultados
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Parallel OSRM task failed: {result}")
                continue
            
            distance, i, j = result
            matrix[i][j] = distance
            matrix[j][i] = distance  # Matriz simétrica
        
        return matrix
    
    async def _get_osrm_distance_with_indices(self, origin: Dict, destination: Dict, 
                                            i: int, j: int) -> Tuple[float, int, int]:
        """Helper para cálculo paralelo con índices"""
        try:
            distance = await self._get_osrm_distance(origin, destination)
            return distance, i, j
        except Exception as e:
            logger.warning(f"OSRM distance {i}→{j} failed: {e}, using geodesic")
            # Fallback geodesic
            geodesic_dist = geodesic(
                (origin['lat'], origin['lon']),
                (destination['lat'], destination['lon'])
            ).kilometers
            return geodesic_dist, i, j
    
    async def _get_osrm_distance(self, origin: Dict, destination: Dict) -> float:
        """Obtener distancia OSRM entre dos puntos"""
        if not self.has_osrm:
            raise Exception("OSRM service not available")
        
        try:
            # Usar el servicio OSRM real existente
            result = await self.osrm_service.get_distance_km(
                lat1=origin['lat'], lon1=origin['lon'],
                lat2=destination['lat'], lon2=destination['lon']
            )
            
            if result and result > 0:
                return result
            else:
                raise Exception(f"OSRM returned invalid distance: {result}")
                
        except Exception as e:
            logger.debug(f"OSRM distance calculation failed: {e}")
            raise e
    
    async def _calculate_geodesic_matrix(self, places: List[Dict]) -> List[List[float]]:
        """Calcular matriz usando distancias geodésicas (rápido, menos preciso)"""
        matrix = []
        
        for i, origin in enumerate(places):
            row = []
            for j, destination in enumerate(places):
                if i == j:
                    row.append(0.0)
                else:
                    distance = geodesic(
                        (origin['lat'], origin['lon']),
                        (destination['lat'], destination['lon'])
                    ).kilometers
                    row.append(distance)
            
            matrix.append(row)
        
        return matrix
    
    async def _save_to_cache(self, places_hash: str, matrix: List[List[float]], 
                           source: str, places: List[Dict]):
        """Guardar matriz en cache con cleanup automático"""
        
        # Cleanup cache si está lleno
        if len(self.cache) >= self.max_cache_size:
            await self._cleanup_cache()
        
        # Crear entrada de cache
        cached_matrix = CachedDistanceMatrix(
            matrix_data=matrix,
            places_hash=places_hash,
            timestamp=datetime.now(),
            source=source,
            cache_ttl=self.cache_ttl,
            metadata={
                "places_count": len(places),
                "matrix_size": f"{len(matrix)}x{len(matrix[0]) if matrix else 0}",
                "total_distances": sum(sum(row) for row in matrix)
            }
        )
        
        self.cache[places_hash] = cached_matrix
        self.stats["cache_size"] = len(self.cache)
        self.stats["total_matrices_cached"] += 1
        
        logger.debug(f"💾 Matrix cached - {len(places)} places, source: {source}")
    
    async def _cleanup_cache(self):
        """Limpiar cache eliminando entradas más antiguas y expiradas"""
        logger.info(f"🧹 Cleaning up cache - current size: {len(self.cache)}")
        
        # Eliminar entradas expiradas
        expired_keys = []
        for key, cached_matrix in self.cache.items():
            if cached_matrix.is_expired:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        # Si aún está lleno, eliminar las más antiguas
        if len(self.cache) >= self.max_cache_size:
            # Ordenar por timestamp y eliminar las más antiguas
            sorted_items = sorted(
                self.cache.items(),
                key=lambda item: item[1].timestamp
            )
            
            # Eliminar el 20% más antiguo
            cleanup_count = max(1, len(sorted_items) // 5)
            for i in range(cleanup_count):
                key = sorted_items[i][0]
                del self.cache[key]
        
        self.stats["cache_size"] = len(self.cache)
        logger.info(f"✅ Cache cleanup completed - new size: {len(self.cache)}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de performance del cache"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "hit_rate": round(hit_rate, 3),
            "total_requests": total_requests,
            "cache_hits": self.stats["hits"],
            "cache_misses": self.stats["misses"],
            "cache_size": self.stats["cache_size"],
            "max_cache_size": self.max_cache_size,
            "avg_calculation_time_ms": round(self.stats["avg_calculation_time_ms"], 1),
            "total_matrices_cached": self.stats["total_matrices_cached"],
            "cache_ttl_seconds": self.cache_ttl,
            "cache_efficiency": "excellent" if hit_rate > 0.8 else "good" if hit_rate > 0.6 else "needs_improvement"
        }
    
    async def invalidate_cache(self, places_hash: Optional[str] = None):
        """Invalidar cache específico o todo el cache"""
        if places_hash:
            if places_hash in self.cache:
                del self.cache[places_hash]
                logger.info(f"🗑️ Cache invalidated for hash: {places_hash}")
        else:
            self.cache.clear()
            self.stats["cache_size"] = 0
            logger.info("🗑️ All cache invalidated")

# Singleton instance
ortools_distance_cache = ORToolsDistanceCache()

# Export functions para fácil uso
async def get_cached_distance_matrix(places: List[Dict], source: str = "auto") -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    Función helper para obtener matriz de distancia con cache
    
    Usage:
        matrix, metadata = await get_cached_distance_matrix(places, "osrm")
        print(f"Matrix calculated in {metadata['calculation_time_ms']:.1f}ms")
    """
    return await ortools_distance_cache.get_distance_matrix(places, source)

def get_distance_cache_stats() -> Dict[str, Any]:
    """Obtener estadísticas del cache de distancias OR-Tools"""
    return ortools_distance_cache.get_cache_stats()

async def invalidate_distance_cache(places_hash: Optional[str] = None):
    """Invalidar cache de distancias"""
    await ortools_distance_cache.invalidate_cache(places_hash)