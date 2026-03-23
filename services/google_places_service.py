from typing import List, Dict, Optional, Any
import logging
import asyncio
from utils.google_maps_client import GoogleMapsClient
from utils.geographic_cache_manager import get_cache_manager
from settings import settings

class GooglePlacesService:
    def __init__(self):
        self.maps_client = GoogleMapsClient()
        self.logger = logging.getLogger(__name__)
        self.api_key = settings.GOOGLE_PLACES_API_KEY
        self.cache_manager = get_cache_manager()
        
        # Estadísticas de caché
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls_saved = 0
    
    async def search_nearby(
        self, 
        lat: float, 
        lon: float, 
        types: Optional[List[str]] = None, 
        radius_m: int = 3000, 
        limit: int = 3,  # Cambiado de 6 a 3
        **kwargs  # Para compatibilidad con parámetros adicionales como 'preferences'
    ) -> List[Dict]:
        """
        🔍 Búsqueda robusta de lugares cercanos con manejo de errores
        SIN CACHÉ - Siempre genera sugerencias frescas
        """
        try:
            # Usar tipos por defecto si no se proporcionan
            if types is None:
                types = ['tourist_attraction', 'museum', 'point_of_interest']
            
            # Filtrar kwargs no reconocidos
            _ = kwargs.pop('preferences', None)  # Ignorar preferences si existe
            
            # Implementar búsqueda con retry
            for attempt in range(2):  # 2 intentos
                try:
                    if hasattr(self.maps_client, 'search_nearby_places'):
                        self.logger.info(f"🔍 [INTENTO {attempt+1}] Llamando search_nearby_places: lat={lat:.4f}, lon={lon:.4f}, types={types}, radius={radius_m}")
                        results = await self.maps_client.search_nearby_places(
                            lat, lon, types, radius_m, limit
                        )
                        self.logger.info(f"📊 [INTENTO {attempt+1}] Google Maps API devolvió: {len(results) if results else 0} resultados")
                        if results and len(results) >= limit:
                            self.logger.info(f"✅ Google Maps client: {len(results)} lugares encontrados - USANDO REALES")
                            return results[:limit]
                        elif results and len(results) > 0:
                            self.logger.warning(f"⚠️ Google Maps devolvió solo {len(results)} lugares de {limit} solicitados - USANDO LOS QUE HAY")
                            return results
                        else:
                            # Si Google Maps no devuelve suficientes resultados, usar sintéticas
                            self.logger.warning(f"⚠️ Google Maps sin resultados (attempt {attempt+1}), usando sintéticas")
                            return self._generate_synthetic_suggestions(lat, lon, types, limit)
                    else:
                        # Fallback: generar sugerencias sintéticas basadas en ubicación
                        self.logger.warning(f"🤖 Método search_nearby_places no disponible, usando sintéticas para {lat:.3f},{lon:.3f}")
                        return self._generate_synthetic_suggestions(lat, lon, types, limit)
                        
                except Exception as e:
                    if attempt == 0:
                        self.logger.warning(f"Primer intento de búsqueda falló: {e}")
                        await asyncio.sleep(1)  # Wait 1 second before retry
                        continue
                    else:
                        raise e
                        
        except Exception as e:
            self.logger.warning(f"Búsqueda de lugares falló: {e}")
            # En caso de error, siempre devolver sugerencias sintéticas
            return self._generate_synthetic_suggestions(lat, lon, types or ['point_of_interest'], limit)
    
    def _generate_synthetic_suggestions(self, lat: float, lon: float, types: List[str], limit: int) -> List[Dict]:
        """Generar exactamente 3 sugerencias sintéticas cuando la API falla"""
        synthetic_places = []
        
        # Base de datos expandida de sugerencias por tipo (más variedad)
        type_suggestions = {
            'restaurant': [
                'Restaurante local', 'Lugar de comida típica', 'Bistró familiar',
                'Comida casera', 'Parrilla tradicional', 'Casa de comidas',
                'Restaurante regional', 'Cocina local', 'Lugar gastronómico'
            ],
            'tourist_attraction': [
                'Sitio histórico', 'Mirador', 'Plaza principal',
                'Monumento local', 'Punto panorámico', 'Lugar emblemático',
                'Atracción cultural', 'Sitio de interés', 'Lugar destacado'
            ],
            'museum': [
                'Centro cultural', 'Galería de arte', 'Museo local',
                'Espacio cultural', 'Museo histórico', 'Centro de arte',
                'Galería local', 'Museo temático', 'Espacio expositivo'
            ],
            'park': [
                'Parque urbano', 'Plaza verde', 'Área recreativa',
                'Espacio verde', 'Parque central', 'Zona natural',
                'Área de descanso', 'Parque local', 'Espacio público'
            ],
            'shopping_mall': [
                'Centro comercial', 'Mercado local', 'Tiendas',
                'Galería comercial', 'Plaza comercial', 'Centro de compras'
            ],
            'cafe': [
                'Café local', 'Lugar de café', 'Cafetería',
                'Café artesanal', 'Casa de té', 'Espacio café'
            ],
            'lodging': ['Hotel Plaza', 'Hotel Centro', 'Hostal Local'],
            'accommodation': ['Hotel Ejecutivo', 'Hotel Boutique', 'Hotel Business'],
            'point_of_interest': [
                'Lugar de interés', 'Punto destacado', 'Sitio relevante',
                'Atracción local', 'Punto turístico', 'Lugar notable'
            ]
        }
        
        # Crear semilla única basada en ubicación para consistencia pero con variación
        import hashlib
        location_hash = hashlib.md5(f"{lat:.4f},{lon:.4f}".encode()).hexdigest()[:8]
        import random
        random.seed(int(location_hash, 16))
        
        # Generar exactamente las sugerencias solicitadas (garantizar al menos 'limit')
        max_suggestions = max(limit, 3)  # Al menos 3, pero puede ser más si se solicita
        
        # Inferir ciudad una sola vez
        city_name = self._infer_city_name(lat, lon)
        
        for i in range(max_suggestions):
            place_type = types[i % len(types)]
            suggestions = type_suggestions.get(place_type, ['Lugar de interés'])
            base_name = suggestions[i % len(suggestions)]
            
            # Crear nombres más variados y específicos
            if place_type in ['lodging', 'accommodation']:
                if city_name and city_name not in base_name:
                    name = f"{base_name} {city_name}"
                else:
                    name = base_name
            else:
                # Agregar variación con modificadores
                modifiers = ['Central', 'del Centro', 'Principal', 'Local', 'Tradicional', 'Histórico']
                modifier = modifiers[i % len(modifiers)]
                name = f"{base_name} {modifier}"
            
            # Coordenadas con distribución más natural (patrón circular)
            import math
            angle = (i * 2 * math.pi) / max_suggestions  # Distribución circular
            radius_offset = 0.003 + (i * 0.001)  # Radio creciente
            
            offset_lat = lat + (radius_offset * math.cos(angle))
            offset_lon = lon + (radius_offset * math.sin(angle))
            
            # Calcular distancia aproximada
            distance_km = self._calculate_distance(lat, lon, offset_lat, offset_lon)
            eta_minutes = max(1, int(distance_km * 1000 / 83.33))  # Mínimo 1 minuto
            
            # Rating más variado pero consistente
            base_rating = 3.8 + (hash(f"{place_type}_{i}") % 10) / 10  # 3.8 a 4.8
            rating = round(base_rating, 1)
            
            synthetic_places.append({
                'name': name,
                'lat': offset_lat,
                'lon': offset_lon,
                'type': place_type,
                'rating': rating,
                'eta_minutes': eta_minutes,
                'reason': f"buen rating ({rating}⭐), {'muy cerca' if eta_minutes < 5 else 'cerca'}",
                'synthetic': True
            })
        
        return synthetic_places
    
    def _infer_city_name(self, lat: float, lon: float) -> str:
        """Inferir nombre de ciudad basándose en coordenadas aproximadas de Chile"""
        # Ciudades principales de Chile con sus coordenadas aproximadas
        cities = [
            (-33.4489, -70.6693, "Santiago"),
            (-23.6509, -70.3975, "Antofagasta"), 
            (-29.9027, -71.2519, "La Serena"),
            (-36.8201, -73.0444, "Concepción"),
            (-39.8142, -73.2459, "Valdivia"),
            (-20.2141, -70.1522, "Iquique"),
            (-27.3668, -70.4037, "Copiapó"),
            (-22.4908, -68.9016, "Calama"),
            (-18.4783, -70.3146, "Arica"),
            (-41.4693, -72.9424, "Puerto Montt"),
            (-53.1638, -70.9171, "Punta Arenas")
        ]
        
        min_distance = float('inf')
        closest_city = ""
        
        for city_lat, city_lon, city_name in cities:
            distance = ((lat - city_lat) ** 2 + (lon - city_lon) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_city = city_name
        
        # Solo devolver ciudad si está relativamente cerca (< 1 grado ~ 100km)
        if min_distance < 1.0:
            return closest_city
        return ""
    
    async def search_nearby_real(
        self,
        lat: float,
        lon: float,
        radius_m: int = 3000,
        types: Optional[List[str]] = None,
        limit: int = 3,
        exclude_chains: bool = True,
        day_offset: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Buscar lugares reales cercanos usando Google Places API con variedad por día
        Con caché inteligente para reducir llamadas API
        """
        try:
            # 🎯 PASO 1: INTENTAR CACHE PRIMERO
            place_types = self._get_types_for_day(types, day_offset)
            cached_results = self.cache_manager.get_cached_places(
                lat=lat, 
                lon=lon, 
                radius=radius_m, 
                place_types=place_types
            )
            
            if cached_results and len(cached_results) >= limit:
                self.cache_hits += 1
                self.api_calls_saved += len(place_types)  # Llamadas API que nos ahorramos
                
                self.logger.info(f"🎯 CACHE HIT: {len(cached_results)} lugares desde caché")
                self.logger.debug(f"   💰 API calls ahorradas: {len(place_types)}")
                
                # Aplicar filtros post-caché
                filtered_cached = []
                for place in cached_results[:limit * 2]:  # Tomar más para filtrar
                    if self._is_valid_suggestion(place, exclude_chains):
                        filtered_cached.append(place)
                        if len(filtered_cached) >= limit:
                            break
                
                return filtered_cached[:limit]
            
            # 🔄 PASO 2: SI NO HAY CACHE VÁLIDO, USAR API
            self.cache_misses += 1
            
            if not self.api_key:
                self.logger.warning("🔑 No hay API key de Google Places - sin sugerencias (solo lugares de alta calidad)")
                return []  # Sin API key no podemos validar calidad, así que no devolvemos nada
            
            # Configurar tipos de búsqueda con rotación por día
            place_types = self._get_types_for_day(types, day_offset)
            self.logger.info(f"🎯 DÍA {day_offset}: Tipos solicitados={types}, Tipos finales={place_types}")
            
            all_places = []
            
            # 🎯 GARANTIZAR ATRACCIONES TURÍSTICAS PRIMERO
            tourist_places = []
            other_places = []
            
            for place_type in place_types:
                try:
                    # Llamada a Google Places Nearby Search
                    places_result = await self._google_nearby_search(
                        lat=lat,
                        lon=lon,
                        radius=radius_m,
                        type=place_type,
                        limit=8  # Buscar más para poder filtrar y variar
                    )
                    
                    if places_result and 'results' in places_result:
                        # Usar day_offset para seleccionar diferentes resultados por día
                        start_idx = (day_offset - 1) % min(len(places_result['results']), 3)
                        
                        for i, place in enumerate(places_result['results'][start_idx:]):
                            processed_place = self._process_google_place(place, lat, lon)
                            if processed_place and self._is_valid_suggestion(processed_place, exclude_chains):
                                # 🎯 Separar por tipo para garantizar atracciones turísticas
                                if place_type == 'tourist_attraction':
                                    tourist_places.append(processed_place)
                                else:
                                    other_places.append(processed_place)
                                
                except Exception as e:
                    self.logger.warning(f"Error searching {place_type}: {e}")
                    continue
            
            # 🎯 COMBINAR RESULTADOS: PRIORIZAR ATRACCIONES TURÍSTICAS
            final_places = []
            
            # Primero agregar atracciones turísticas (al menos 1 si existe)
            if tourist_places:
                import random
                random.seed(day_offset * 42)  # Seed basado en día para consistencia
                sorted_tourist = sorted(tourist_places, key=lambda x: (-x['rating'] + random.uniform(-0.1, 0.1), x['eta_minutes']))
                final_places.extend(sorted_tourist[:2])  # Máximo 2 atracciones turísticas
            
            # Luego agregar otros tipos para variedad
            if other_places:
                import random
                random.seed(day_offset * 73)  # Seed diferente para otros tipos
                sorted_others = sorted(other_places, key=lambda x: (-x['rating'] + random.uniform(-0.1, 0.1), x['eta_minutes']))
                remaining_slots = limit - len(final_places)
                final_places.extend(sorted_others[:remaining_slots])
            
            # Si no hay resultados reales de calidad, devolver vacío
            if not final_places:
                self.logger.info("🚫 Sin lugares que cumplan estándares de calidad (4.5⭐, 20+ reseñas)")
                return []
            
            # 🎯 PASO 3: CACHEAR RESULTADOS PARA FUTURAS BÚSQUEDAS
            self.cache_manager.cache_places(
                lat=lat,
                lon=lon, 
                radius=radius_m,
                place_types=place_types,
                places_data=final_places
            )
            
            self.logger.info(f"✅ Retornando {len(final_places)} sugerencias (🏛️ {len([p for p in final_places if p.get('type') == 'tourist_attraction'])} atracciones)")
            self.logger.debug(f"💾 Resultados cacheados para futuras búsquedas")
            return final_places[:limit]
            
        except Exception as e:
            self.logger.error(f"❌ Error en búsqueda nearby real: {e}")
            return []  # En caso de error, no devolver sugerencias para mantener calidad

    async def _google_nearby_search(
        self,
        lat: float,
        lon: float,
        radius: int,
        types: Optional[List[str]] = None,
        type: Optional[str] = None,  # Mantener compatibilidad con versión anterior
        limit: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Llamada real a Google Places Nearby Search API"""
        try:
            import aiohttp
            
            # Determinar el tipo a usar
            search_type = type if type else (types[0] if types and len(types) > 0 else 'tourist_attraction')
            
            url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            params = {
                'location': f"{lat},{lon}",
                'radius': radius,
                'type': search_type,
                'key': self.api_key,
                'language': 'es'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('status') == 'OK':
                            self.logger.info(f"✅ Google Places: {len(data.get('results', []))} lugares encontrados para {search_type}")
                            return data
                        else:
                            self.logger.warning(f"Google Places status: {data.get('status')} para {search_type}")
                            return None
                    else:
                        self.logger.warning(f"Google Places HTTP error: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error en Google Places API: {e}")
            return None

    def _get_types_for_day(self, types: Optional[List[str]], day_offset: int) -> List[str]:
        """Obtener tipos simples: SIEMPRE tourist_attraction + variedad"""
        
        if types:
            return types  # Si se especifican tipos específicos, usarlos
        
        # 🎯 ENFOQUE SIMPLE: Siempre tourist_attraction + variedad por día
        variety_types = ['cafe', 'restaurant', 'museum', 'park', 'point_of_interest']
        day_index = (day_offset - 1) % len(variety_types)
        secondary_type = variety_types[day_index]
        
        # SIEMPRE incluir tourist_attraction como primer tipo
        return ['tourist_attraction', secondary_type, 'cafe']

    async def _search_nearby_with_day_variety(
        self,
        lat: float,
        lon: float,
        types: Optional[List[str]],
        radius_m: int,
        limit: int,
        day_offset: int
    ) -> List[Dict[str, Any]]:
        """Fallback con variedad por día para lugares sintéticos"""
        try:
            # Usar tipos específicos por día
            place_types = self._get_types_for_day(types, day_offset)
            
            suggestions = []
            
            # Nombres variados por día y tipo
            name_variations = {
                'restaurant': [
                    ['Restaurante local', 'Bistró familiar', 'Lugar de comida típica'],
                    ['Café gastronómico', 'Restaurante tradicional', 'Casa de comidas'],
                    ['Parrilla local', 'Comida casera', 'Restaurante del barrio']
                ],
                'tourist_attraction': [
                    ['Sitio histórico', 'Mirador', 'Plaza principal'],
                    ['Monumento local', 'Punto panorámico', 'Lugar emblemático'],
                    ['Atracción cultural', 'Sitio de interés', 'Lugar destacado']
                ],
                'museum': [
                    ['Centro cultural', 'Galería de arte', 'Museo local'],
                    ['Espacio cultural', 'Museo histórico', 'Centro de arte'],
                    ['Galería local', 'Museo temático', 'Espacio expositivo']
                ],
                'park': [
                    ['Parque urbano', 'Plaza verde', 'Área recreativa'],
                    ['Espacio verde', 'Parque central', 'Zona natural'],
                    ['Área de descanso', 'Parque local', 'Espacio público']
                ],
                'cafe': [
                    ['Café local', 'Lugar de café', 'Cafetería'],
                    ['Café artesanal', 'Casa de té', 'Espacio café'],
                    ['Café urbano', 'Lugar de encuentro', 'Café típico']
                ],
                'shopping_mall': [
                    ['Centro comercial', 'Mercado local', 'Tiendas'],
                    ['Galería comercial', 'Plaza comercial', 'Centro de compras'],
                    ['Mercado central', 'Zona comercial', 'Centro urbano']
                ],
                'church': [
                    ['Iglesia histórica', 'Templo local', 'Basílica'],
                    ['Capilla', 'Santuario', 'Iglesia colonial'],
                    ['Catedral', 'Templo religioso', 'Iglesia antigua']
                ],
                'art_gallery': [
                    ['Galería de arte', 'Espacio artístico', 'Centro de arte'],
                    ['Galería local', 'Exposición artística', 'Espacio cultural'],
                    ['Galería urbana', 'Centro creativo', 'Espacio de arte']
                ]
            }
            
            for i in range(limit):
                place_type = place_types[i % len(place_types)]
                
                # Seleccionar variación según día
                day_idx = (day_offset - 1) % 3
                type_names = name_variations.get(place_type, [['Lugar de interés']])
                if day_idx < len(type_names):
                    available_names = type_names[day_idx]
                else:
                    available_names = type_names[0]
                
                name = available_names[i % len(available_names)]
                
                # Coordenadas con offset diferente por día
                day_offset_factor = (day_offset - 1) * 0.003  # Más separación entre días
                base_offset = i * 0.002
                offset_lat = lat + base_offset + day_offset_factor
                offset_lon = lon + base_offset + day_offset_factor
                
                # Calcular distancia y ETA
                distance_km = self._calculate_distance(lat, lon, offset_lat, offset_lon)
                eta_minutes = max(0, int(distance_km * 1000 / 83.33))
                
                # Rating progresivo variado por día
                base_rating = 4.0 + (day_offset - 1) * 0.1
                rating = round(base_rating + (i * 0.1), 1)
                
                suggestion = {
                    'name': name,
                    'lat': offset_lat,
                    'lon': offset_lon,
                    'type': place_type,
                    'rating': min(rating, 5.0),
                    'eta_minutes': eta_minutes,
                    'reason': f"buen rating ({rating}⭐), {'muy cerca' if eta_minutes < 5 else 'cerca'}",
                    'synthetic': True,
                    'day_generated': day_offset
                }
                
                suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"❌ Error generando sugerencias sintéticas con variedad: {e}")
            return []

    def _process_google_place(self, place: Dict[str, Any], origin_lat: float, origin_lon: float) -> Optional[Dict[str, Any]]:
        """Procesar lugar de Google Places API"""
        try:
            location = place.get('geometry', {}).get('location', {})
            lat = location.get('lat')
            lon = location.get('lng')
            
            if not lat or not lon:
                return None
            
            # Calcular distancia y ETA
            distance_km = self._calculate_distance(origin_lat, origin_lon, lat, lon)
            eta_minutes = int(distance_km * 1000 / 83.33)  # 5 km/h walking speed
            
            # Obtener tipo principal
            place_types = place.get('types', [])
            main_type = self._get_main_type(place_types)
            
            return {
                'name': place.get('name', 'Lugar de interés'),
                'lat': lat,
                'lon': lon,
                'type': main_type,
                'rating': place.get('rating', 4.0),
                'eta_minutes': eta_minutes,
                'distance_km': distance_km,
                'price_level': place.get('price_level'),
                'user_ratings_total': place.get('user_ratings_total', 0),
                'vicinity': place.get('vicinity', ''),
                'place_id': place.get('place_id'),
                'photos': place.get('photos', []),
                'reason': f"Google Places: {place.get('rating', 4.0)}⭐, {eta_minutes}min caminando",
                'synthetic': False,  # Lugar real
                'source': 'google_places'
            }
            
        except Exception as e:
            self.logger.error(f"Error procesando lugar de Google: {e}")
            return None

    def _get_main_type(self, types: List[str]) -> str:
        """Obtener el tipo principal de un lugar"""
        priority_types = [
            'restaurant', 'tourist_attraction', 'museum', 'park', 
            'shopping_mall', 'cafe', 'bar', 'lodging', 'church'
        ]
        
        for priority_type in priority_types:
            if priority_type in types:
                return priority_type
        
        return types[0] if types else 'point_of_interest'

    def _is_valid_suggestion(self, place: Dict[str, Any], exclude_chains: bool = True) -> bool:
        """Validar si un lugar es una buena sugerencia con filtros de calidad estrictos"""
        try:
            # ⭐ FILTROS DE CALIDAD ESTRICTOS
            
            # 1. Rating mínimo: 4.5 estrellas
            rating = place.get('rating', 0)
            if rating < 4.5:
                self.logger.debug(f"🚫 {place.get('name', 'Lugar')} descartado: rating {rating} < 4.5")
                return False
            
            # 2. Número mínimo de reseñas: 20
            user_ratings_total = place.get('user_ratings_total', 0)
            if user_ratings_total < 20:
                self.logger.debug(f"🚫 {place.get('name', 'Lugar')} descartado: {user_ratings_total} reseñas < 20")
                return False
            
            # 3. Filtrar cadenas conocidas si se solicita
            if exclude_chains:
                chain_keywords = ['mcdonalds', 'kfc', 'burger king', 'subway', 'pizza hut', 'starbucks', 'dominos']
                name_lower = place['name'].lower()
                if any(chain in name_lower for chain in chain_keywords):
                    self.logger.debug(f"🚫 {place.get('name', 'Lugar')} descartado: es una cadena")
                    return False
            
            # 4. Validar distancia máxima (5km)
            if place.get('distance_km', 0) > 5:
                self.logger.debug(f"🚫 {place.get('name', 'Lugar')} descartado: distancia {place.get('distance_km', 0):.1f}km > 5km")
                return False
            
            # ✅ Lugar válido con alta calidad
            self.logger.debug(f"✅ {place.get('name', 'Lugar')} válido: {rating}⭐ ({user_ratings_total} reseñas)")
            return True
            
        except Exception as e:
            self.logger.warning(f"Error validando sugerencia: {e}")
            return False
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcular distancia entre dos puntos usando fórmula haversine"""
        import math
        
        # Radio de la Tierra en km
        R = 6371.0
        
        # Convertir a radianes
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Diferencias
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Fórmula haversine
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de uso del caché"""
        cache_stats = self.cache_manager.get_cache_stats()
        
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        estimated_cost_saved = self.api_calls_saved * 0.032  # $0.032 per API call
        
        return {
            'cache_performance': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate_percentage': round(hit_rate, 2),
                'api_calls_saved': self.api_calls_saved,
                'estimated_cost_saved_usd': round(estimated_cost_saved, 3)
            },
            'cache_storage': cache_stats
        }
    
    def reset_stats(self) -> None:
        """Resetear estadísticas de caché"""
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls_saved = 0
        self.logger.info("📊 Estadísticas de caché reseteadas")
        
        """
        Busca lugares cercanos usando Google Places API.
        
        Args:
            lat: Latitud del centro de búsqueda
            lon: Longitud del centro de búsqueda
            radius: Radio de búsqueda en metros (default 5km)
            place_types: Lista de tipos de lugares a buscar. Si no se especifica,
                        usa una lista predeterminada de tipos turísticos
        
        Returns:
            Lista de lugares encontrados con su información completa
        """
        if place_types is None:
            place_types = [
                'tourist_attraction',
                'museum',
                'park',
                'point_of_interest',
                'natural_feature',
                'amusement_park',
                'art_gallery',
                'aquarium',
                'church',
                'city_hall',
                'historic_site'
            ]
        
        all_places = []
        for place_type in place_types:
            try:
                places = self.maps_client.search_nearby_places(
                    latitude=lat,
                    longitude=lon,
                    radius=radius,
                    place_type=place_type
                )
                if places:
                    all_places.extend(places)
            except Exception as e:
                logging.warning(f"Error buscando lugares de tipo {place_type}: {str(e)}")
                continue
        
        return all_places
    
    def categorize_places(self, places: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Categoriza lugares en tres categorías principales usando sus tipos y atributos.
        
        Args:
            places: Lista de lugares de Google Places API
        
        Returns:
            Diccionario con lugares categorizados en nature_escape, cultural_immersion, adventure_day
        """
        categories = {
            'nature_escape': [],
            'cultural_immersion': [],
            'adventure_day': []
        }
        
        nature_types = {'park', 'natural_feature', 'campground', 'beach', 'hiking_area'}
        cultural_types = {'museum', 'art_gallery', 'church', 'historic_site', 'city_hall'}
        adventure_types = {'amusement_park', 'aquarium', 'zoo', 'stadium', 'sports_complex'}
        
        for place in places:
            place_types = set(place.get('types', []))
            
            # Calcular puntuación para cada categoría
            nature_score = len(place_types.intersection(nature_types))
            cultural_score = len(place_types.intersection(cultural_types))
            adventure_score = len(place_types.intersection(adventure_types))
            
            # Asignar a la categoría con mayor puntuación
            max_score = max(nature_score, cultural_score, adventure_score)
            if max_score == 0:
                # Si no hay match directo, usar heurísticas adicionales
                if 'point_of_interest' in place_types:
                    if place.get('rating', 0) >= 4.5:
                        categories['cultural_immersion'].append(place)
                    else:
                        categories['adventure_day'].append(place)
            else:
                if max_score == nature_score:
                    categories['nature_escape'].append(place)
                elif max_score == cultural_score:
                    categories['cultural_immersion'].append(place)
                else:
                    categories['adventure_day'].append(place)
        
        return categories
    
    def get_transport_options(self, places: List[Dict]) -> Dict[str, str]:
        """
        Determina las mejores opciones de transporte basadas en la distribución de lugares.
        
        Args:
            places: Lista de lugares a analizar
        
        Returns:
            Diccionario con recomendaciones de transporte por categoría
        """
        def calculate_average_distance(coords_list):
            if len(coords_list) < 2:
                return 0
            
            total_distance = 0
            count = 0
            for i in range(len(coords_list)):
                for j in range(i + 1, len(coords_list)):
                    lat1, lon1 = coords_list[i]
                    lat2, lon2 = coords_list[j]
                    distance = self.maps_client.calculate_distance(lat1, lon1, lat2, lon2)
                    total_distance += distance
                    count += 1
            
            return total_distance / count if count > 0 else 0
        
        # Extraer coordenadas de todos los lugares
        coords = [(p['geometry']['location']['lat'], p['geometry']['location']['lng']) 
                 for p in places if 'geometry' in p and 'location' in p['geometry']]
        
        avg_distance = calculate_average_distance(coords)
        
        # Determinar recomendaciones basadas en la distancia promedio
        if avg_distance > 10:  # Si los lugares están a más de 10km en promedio
            return {
                'nature_escape': 'Tour organizado o auto recomendado',
                'cultural_immersion': 'Transporte público o taxi',
                'adventure_day': 'Tour organizado o transporte público'
            }
        elif avg_distance > 3:  # Si los lugares están entre 3-10km
            return {
                'nature_escape': 'Transporte público o auto',
                'cultural_immersion': 'Transporte público',
                'adventure_day': 'Transporte público o bicicleta'
            }
        else:  # Si los lugares están cercanos
            return {
                'nature_escape': 'Transporte público o caminando',
                'cultural_immersion': 'A pie o bicicleta',
                'adventure_day': 'A pie o transporte público'
            }
    
    def generate_day_suggestions(self, lat: float, lon: float) -> Dict[str, List[Dict]]:
        """
        Genera sugerencias completas para un día basadas en la ubicación.
        
        Args:
            lat: Latitud del centro de búsqueda
            lon: Longitud del centro de búsqueda
        
        Returns:
            Diccionario con sugerencias categorizadas y transporte recomendado
        """
        # Buscar lugares cercanos
        places = self.get_nearby_places(lat, lon)
        
        # Categorizar lugares
        categorized_places = self.categorize_places(places)
        
        # Obtener recomendaciones de transporte
        transport_options = self.get_transport_options(places)
        
        # Formatear sugerencias
        suggestions = {}
        for category in categorized_places:
            category_places = categorized_places[category][:4]  # Top 4 lugares por categoría
            
            suggestions[category] = {
                'suggestions': [place['name'] for place in category_places],
                'transport': transport_options[category],
                'places': category_places  # Incluir información completa de los lugares
            }
        
        return suggestions
