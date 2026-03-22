from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Union, Literal, Dict
from datetime import date, time, datetime
from enum import Enum

class PlaceType(str, Enum):
    # Tipos básicos
    RESTAURANT = "restaurant"
    CAFE = "cafe"
    BAR = "bar"
    
    # Atracciones y puntos de interés
    ATTRACTION = "attraction"
    MUSEUM = "museum"
    PARK = "park"
    CHURCH = "church"
    MONUMENT = "monument"
    VIEWPOINT = "viewpoint"
    BEACH = "beach"
    ZOO = "zoo"
    
    # Shopping y entretenimiento
    SHOPPING = "shopping"
    SHOPPING_MALL = "shopping_mall"
    STORE = "store"
    NIGHT_CLUB = "night_club"
    MOVIE_THEATER = "movie_theater"
    
    # Lugares al aire libre
    NATURAL_FEATURE = "natural_feature"
    POINT_OF_INTEREST = "point_of_interest"
    
    # Otros tipos comunes de Google Places
    LODGING = "lodging"
    ACCOMMODATION = "accommodation"
    FOOD = "food"
    ESTABLISHMENT = "establishment"
    ART_GALLERY = "art_gallery"
    TOURIST_ATTRACTION = "tourist_attraction"

class TransportMode(str, Enum):
    WALK = "walk"
    DRIVE = "drive"
    TRANSIT = "transit"
    BIKE = "bike"

class Coordinates(BaseModel):
    """Coordenadas geográficas"""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

class Place(BaseModel):
    id: Optional[str] = None
    name: str = Field(..., min_length=1, max_length=100)
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)  # Campo primario para longitud
    long: Optional[float] = None  # Alias opcional para longitud
    type: Optional[PlaceType] = None  # Campo primario para tipo
    category: Optional[PlaceType] = None  # Alias opcional para tipo
    priority: Optional[int] = Field(default=5, ge=1, le=10)
    min_duration_hours: Optional[float] = Field(default=None, ge=0.5, le=8)
    opening_hours: Optional[str] = None
    rating: Optional[float] = Field(default=None, ge=0, le=5)
    image: Optional[str] = None
    address: Optional[str] = None
    google_place_id: Optional[str] = None

    @field_validator('name')
    @classmethod
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('El nombre no puede estar vacío')
        return v.strip()

    @field_validator('lon', 'long', mode='before')
    @classmethod
    def validate_longitude(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError('La longitud debe ser un número válido')
        return v

    @field_validator('lat', mode='before')
    @classmethod
    def validate_latitude(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError('La latitud debe ser un número válido')
        return v

    @field_validator('type', 'category', mode='before')
    @classmethod
    def validate_type(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            # Mapear categorías comunes a tipos válidos
            category_mapping = {
                'restaurant': PlaceType.RESTAURANT,
                'accommodation': PlaceType.ACCOMMODATION,
                'shopping': PlaceType.SHOPPING,
                'attraction': PlaceType.TOURIST_ATTRACTION,
                'lodging': PlaceType.LODGING,
                'cafe': PlaceType.CAFE,
                'bar': PlaceType.BAR,
                'store': PlaceType.STORE,
                'movie_theater': PlaceType.MOVIE_THEATER,
                'museum': PlaceType.MUSEUM,
                'park': PlaceType.PARK,
                'church': PlaceType.CHURCH,
                'monument': PlaceType.MONUMENT,
                'beach': PlaceType.BEACH,
                'zoo': PlaceType.ZOO,
                'night_club': PlaceType.NIGHT_CLUB,
                'shopping_mall': PlaceType.SHOPPING_MALL,
                'point_of_interest': PlaceType.POINT_OF_INTEREST,
                'tourist_attraction': PlaceType.TOURIST_ATTRACTION,
                'establishment': PlaceType.ESTABLISHMENT,
                'food': PlaceType.FOOD
            }
            normalized_type = v.lower().replace(' ', '_')
            return category_mapping.get(normalized_type, PlaceType.POINT_OF_INTEREST)
        return v

    @model_validator(mode='before')
    def check_longitude_and_type(cls, values):
        if isinstance(values, dict):
            # Manejar longitud (lon/long)
            lon = values.get('lon')
            long = values.get('long')
            if lon is None and long is not None:
                values['lon'] = long
            elif lon is not None and long is None:
                values['long'] = lon

            # Manejar tipo (type/category)
            type_val = values.get('type')
            category_val = values.get('category')
            if type_val is None and category_val is not None:
                values['type'] = category_val
            elif type_val is not None and category_val is None:
                values['category'] = type_val

        return values

    def get_longitude(self) -> float:
        """Obtener la longitud desde lon o long"""
        return self.lon if self.lon is not None else self.long

    def get_type(self) -> str:
        """Obtener el tipo desde type o category"""
        return self.type.value if self.type is not None else (self.category.value if self.category is not None else None)

    class Config:
        populate_by_name = True

class DailySchedule(BaseModel):
    """Horarios personalizados para un día específico"""
    date: str = Field(..., description="Fecha en formato YYYY-MM-DD")
    start_hour: int = Field(..., ge=0, le=23, description="Hora de inicio del día (0-23)")
    end_hour: int = Field(..., ge=0, le=23, description="Hora de fin del día (0-23)")
    
    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('La fecha debe estar en formato YYYY-MM-DD')
    
    @field_validator('end_hour')
    @classmethod
    def validate_hours(cls, v, info):
        if 'start_hour' in info.data and v <= info.data['start_hour']:
            raise ValueError('end_hour debe ser mayor que start_hour')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "date": "2025-11-10",
                "start_hour": 11,
                "end_hour": 18
            }
        }

class Activity(BaseModel):
    place: str
    start: str
    end: str
    duration_h: float
    lat: float
    lon: float
    type: str
    name: str
    category: str
    estimated_duration: float
    priority: int
    coordinates: Coordinates

class ItineraryRequest(BaseModel):
    places: List[Place] = Field(..., min_items=0, max_items=50)  # 🆕 Permitir días libres (min_items=0)
    start_date: Union[date, str]
    end_date: Union[date, str]
    transport_mode: Union[TransportMode, str] = TransportMode.WALK
    daily_start_hour: int = Field(default=9, ge=6, le=12, description="Hora de inicio por defecto para todos los días")
    daily_end_hour: int = Field(default=18, ge=15, le=23, description="Hora de fin por defecto para todos los días")
    custom_schedules: Optional[List[DailySchedule]] = Field(default=None, description="Horarios personalizados para días específicos")
    max_walking_distance_km: Optional[float] = Field(default=15.0, ge=1, le=50)
    max_daily_activities: int = Field(default=6, ge=1, le=10)
    preferences: Optional[Dict] = Field(default_factory=dict)
    accommodations: Optional[List[Place]] = Field(default_factory=list)

    @field_validator('transport_mode', mode='before')
    @classmethod
    def validate_transport_mode(cls, v):
        if isinstance(v, str):
            v = v.strip('"').lower()
            return TransportMode(v)
        return v

    @field_validator('start_date', 'end_date', mode='before')
    @classmethod
    def validate_dates(cls, v):
        if isinstance(v, str):
            try:
                return datetime.strptime(v, '%Y-%m-%d').date()
            except ValueError as e:
                raise ValueError(f'Formato de fecha inválido. Debe ser YYYY-MM-DD: {str(e)}')
        return v
    
    @field_validator('end_date')
    @classmethod
    def end_date_after_start(cls, v, info):
        if 'start_date' in info.data:
            if v < info.data['start_date']:
                raise ValueError('Fecha fin debe ser posterior a fecha inicio')
            trip_days = (v - info.data['start_date']).days
            if trip_days > 90:
                raise ValueError('Duracion maxima del viaje es 90 dias')
        return v

    class Config:
        validate_assignment = True
        extra = "ignore"

class ItineraryResponse(BaseModel):
    itinerary: List[Dict]
    optimization_metrics: Dict
    recommendations: List[str]

class HotelRecommendationRequest(BaseModel):
    places: List[Place]
    max_recommendations: Optional[int] = Field(default=5, ge=1, le=20)
    price_preference: Optional[str] = "any"  # "budget", "mid", "luxury", "any"

    class Config:
        validate_assignment = True
        extra = "ignore"

class PlaceSuggestion(BaseModel):
    suggestions: List[str]
    transport: str
    places: List[Dict]
    
    @classmethod
    def from_new_format(cls, data: Dict):
        """Convert new format suggestions to old format"""
        return cls(
            suggestions=data.get('suggestions', []),
            transport=data.get('transport', 'No especificado'),
            places=data.get('places', [])
        )

class PlaceSuggestionResponse(BaseModel):
    nature_escape: PlaceSuggestion
    cultural_immersion: PlaceSuggestion
    adventure_day: PlaceSuggestion
    performance: Optional[Dict] = None

# ===== MULTI-CIUDAD SCHEMAS =====

class MultiCityOptimizationRequest(BaseModel):
    """Request para optimización multi-ciudad"""
    places: List[Place] = Field(..., description="Lista de POIs a visitar")
    duration_days: int = Field(..., gt=0, le=30, description="Duración del viaje en días")
    start_city: Optional[str] = Field(None, description="Ciudad de inicio preferida")
    optimization_level: Literal["fast", "balanced", "thorough"] = Field("balanced")
    include_accommodations: bool = Field(True, description="Incluir recomendaciones de hoteles")
    budget_level: Literal["budget", "mid_range", "luxury"] = Field("mid_range")
    
class CityInfo(BaseModel):
    """Información de una ciudad en el itinerario"""
    name: str
    country: str
    coordinates: Coordinates
    pois_count: int
    assigned_days: int
    
class AccommodationInfo(BaseModel):
    """Información de accommodation"""
    city: str
    hotel_name: str
    rating: float
    price_range: str
    nights: int
    check_in_day: int
    check_out_day: int
    estimated_cost_usd: float
    coordinates: Coordinates

class MultiCityItineraryResponse(BaseModel):
    """Response completa de itinerario multi-ciudad"""
    success: bool
    cities: List[CityInfo]
    city_sequence: List[str] = Field(..., description="Secuencia optimizada de ciudades")
    daily_schedule: Dict[int, List[Place]] = Field(..., description="Schedule día por día")
    accommodations: List[AccommodationInfo] = Field(default_factory=list)
    
    # Métricas del viaje
    total_duration_days: int
    countries_count: int
    total_distance_km: float
    estimated_accommodation_cost_usd: float
    
    # Metadata de optimización
    optimization_strategy: str
    confidence: float
    processing_time_ms: float
    
    # Análisis logístico
    logistics: Dict = Field(default_factory=dict, description="Análisis de complejidad logística")
    
class MultiCityAnalysisRequest(BaseModel):
    """Request para análisis de viabilidad multi-ciudad"""
    places: List[Place]
    
class MultiCityAnalysisResponse(BaseModel):
    """Response de análisis multi-ciudad"""
    cities_detected: int
    countries_detected: int  
    max_intercity_distance_km: float
    complexity_level: Literal["simple", "intercity", "international", "complex", "international_complex"]
    recommended_duration_days: int
    optimization_recommendation: str
    feasibility_score: float = Field(..., ge=0, le=1)
    warnings: List[str] = Field(default_factory=list)
