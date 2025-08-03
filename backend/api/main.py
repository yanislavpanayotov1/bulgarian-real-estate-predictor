from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import asyncio
from datetime import datetime
import json

# Import our ML model
import sys
sys.path.append('.')
sys.path.append('ml_model')
from ml_model.train_model import RealEstatePricePredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bulgarian Real Estate Price Predictor API",
    description="API for predicting Bulgarian real estate prices using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
predictor = None

# Pydantic models for request/response validation
class PropertyInput(BaseModel):
    """Input model for property price prediction"""
    size: float = Field(..., gt=0, le=500, description="Property size in square meters")
    rooms: int = Field(..., ge=1, le=10, description="Number of rooms")
    floor: int = Field(..., ge=1, le=50, description="Floor number")
    year_built: int = Field(..., ge=1800, le=2024, description="Year the property was built")
    city: str = Field(..., description="City name")
    neighborhood: Optional[str] = Field(None, description="Neighborhood name")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude coordinate")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude coordinate")
    features: Optional[List[str]] = Field(default=[], description="Property features/amenities")
    
    class Config:
        schema_extra = {
            "example": {
                "size": 75.0,
                "rooms": 2,
                "floor": 3,
                "year_built": 2010,
                "city": "Sofia",
                "neighborhood": "Center",
                "latitude": 42.7,
                "longitude": 23.3,
                "features": ["parking", "balcony", "elevator"]
            }
        }

class PricePrediection(BaseModel):
    """Response model for price prediction"""
    predicted_price: float = Field(..., description="Predicted price in BGN")
    confidence_lower: float = Field(..., description="Lower bound of confidence interval")
    confidence_upper: float = Field(..., description="Upper bound of confidence interval")
    price_per_sqm: float = Field(..., description="Predicted price per square meter")
    model_used: str = Field(..., description="Name of the ML model used")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction")

class PropertyData(BaseModel):
    """Model for property data in listings"""
    id: Optional[str] = None
    price: Optional[float] = None
    size: Optional[float] = None
    rooms: Optional[int] = None
    floor: Optional[int] = None
    year_built: Optional[int] = None
    city: Optional[str] = None
    neighborhood: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    features: Optional[List[str]] = []
    url: Optional[str] = None
    description: Optional[str] = None

class MarketStats(BaseModel):
    """Model for market statistics"""
    avg_price: float
    avg_price_per_sqm: float
    avg_size: float
    total_properties: int
    price_range: Dict[str, float]
    popular_cities: List[Dict[str, Any]]
    property_age_distribution: Dict[str, int]

class NeighborhoodInfo(BaseModel):
    """Model for neighborhood information"""
    name: str
    city: str
    avg_price: float
    avg_price_per_sqm: float
    property_count: int
    latitude: float
    longitude: float

# Startup event to load the model
@app.on_event("startup")
async def startup_event():
    """Load the trained ML model on startup"""
    global predictor
    predictor = RealEstatePricePredictor()
    
    # Try to load existing model
    model_loaded = predictor.load_model("../models")
    
    if not model_loaded:
        logger.info("No trained model found. Training new model...")
        # Train a new model if none exists
        try:
            df = predictor.load_data()
            df = predictor.feature_engineering(df)
            X, y = predictor.prepare_features(df)
            predictor.train_models(X, y)
            predictor.save_model("../models")
            logger.info("New model trained and saved successfully")
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            raise e
    else:
        logger.info("Model loaded successfully")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": predictor is not None
    }

# Main prediction endpoint
@app.post("/predict", response_model=PricePrediection)
async def predict_price(property_input: PropertyInput):
    """Predict property price based on input features"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to dictionary
        property_data = property_input.dict()
        
        # Handle coordinates
        if property_input.latitude and property_input.longitude:
            property_data['coordinates'] = {
                'lat': property_input.latitude,
                'lng': property_input.longitude
            }
        
        # Make prediction
        prediction = predictor.predict(property_data)
        
        # Calculate price per square meter
        price_per_sqm = prediction['predicted_price'] / property_input.size
        
        return PricePrediection(
            predicted_price=prediction['predicted_price'],
            confidence_lower=prediction['confidence_lower'],
            confidence_upper=prediction['confidence_upper'],
            price_per_sqm=price_per_sqm,
            model_used=prediction['model_used'],
            prediction_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Get property listings
@app.get("/properties", response_model=List[PropertyData])
async def get_properties(
    city: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    limit: int = 100
):
    """Get property listings with optional filters"""
    try:
        # Load property data - try multiple possible paths
        possible_paths = [
            "../data/raw_properties.csv",
            "data/raw_properties.csv", 
            "../../data/raw_properties.csv"
        ]
        
        df = None
        for data_path in possible_paths:
            if Path(data_path).exists():
                df = pd.read_csv(data_path)
                break
        
        if df is None:
            # Create sample data if no real data exists
            df = predictor._create_sample_data() if predictor else pd.DataFrame()
        
        # Apply filters
        if city:
            df = df[df['city'].str.contains(city, case=False, na=False)]
        if min_price:
            df = df[df['price'] >= min_price]
        if max_price:
            df = df[df['price'] <= max_price]
        if min_size:
            df = df[df['size'] >= min_size]
        if max_size:
            df = df[df['size'] <= max_size]
        
        # Limit results
        df = df.head(limit)
        
        # Convert to response format
        properties = []
        for _, row in df.iterrows():
            # Parse coordinates
            lat, lng = None, None
            if 'coordinates' in row and row['coordinates']:
                try:
                    if isinstance(row['coordinates'], str):
                        coords = eval(row['coordinates'])
                        lat = coords.get('lat')
                        lng = coords.get('lng')
                    elif isinstance(row['coordinates'], dict):
                        lat = row['coordinates'].get('lat')
                        lng = row['coordinates'].get('lng')
                except:
                    # Default coordinates for Bulgarian cities
                    lat, lng = 42.7, 23.3
            
            # Parse features
            features = []
            if row.get('features'):
                try:
                    if isinstance(row['features'], str):
                        features = eval(row['features'])
                    elif isinstance(row['features'], list):
                        features = row['features']
                except:
                    features = []
            
            prop_data = PropertyData(
                price=row.get('price'),
                size=row.get('size'),
                rooms=row.get('rooms'),
                floor=row.get('floor'),
                year_built=row.get('year_built'),
                city=row.get('city'),
                neighborhood=row.get('neighborhood'),
                latitude=lat,
                longitude=lng,
                features=features,
                url=row.get('url'),
                description=row.get('description')
            )
            properties.append(prop_data)
        
        return properties
        
    except Exception as e:
        logger.error(f"Error fetching properties: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch properties: {str(e)}")

# Market statistics endpoint
@app.get("/market-stats", response_model=MarketStats)
async def get_market_stats():
    """Get overall market statistics"""
    try:
        # Load property data - try multiple possible paths
        possible_paths = [
            "../data/raw_properties.csv",
            "data/raw_properties.csv", 
            "../../data/raw_properties.csv"
        ]
        
        df = None
        for data_path in possible_paths:
            if Path(data_path).exists():
                df = pd.read_csv(data_path)
                break
        
        if df is None:
            df = predictor._create_sample_data() if predictor else pd.DataFrame()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No property data available")
        
        # Calculate statistics
        avg_price = float(df['price'].mean())
        avg_size = float(df['size'].mean())
        avg_price_per_sqm = avg_price / avg_size
        total_properties = len(df)
        
        # Price range
        price_range = {
            'min': float(df['price'].min()),
            'max': float(df['price'].max()),
            'q25': float(df['price'].quantile(0.25)),
            'q75': float(df['price'].quantile(0.75))
        }
        
        # Popular cities
        popular_cities = df.groupby('city').agg({
            'price': ['count', 'mean']
        }).round(2).reset_index()
        popular_cities.columns = ['city', 'count', 'avg_price']
        popular_cities = popular_cities.sort_values('count', ascending=False).head(10)
        popular_cities_list = [
            {'city': row['city'], 'count': int(row['count']), 'avg_price': float(row['avg_price'])}
            for _, row in popular_cities.iterrows()
        ]
        
        # Property age distribution
        current_year = datetime.now().year
        df['age_category'] = pd.cut(
            current_year - df['year_built'],
            bins=[0, 5, 15, 30, 100],
            labels=['New (0-5y)', 'Recent (5-15y)', 'Mature (15-30y)', 'Old (30y+)']
        )
        age_distribution = df['age_category'].value_counts().to_dict()
        age_distribution = {str(k): int(v) for k, v in age_distribution.items()}
        
        return MarketStats(
            avg_price=avg_price,
            avg_price_per_sqm=avg_price_per_sqm,
            avg_size=avg_size,
            total_properties=total_properties,
            price_range=price_range,
            popular_cities=popular_cities_list,
            property_age_distribution=age_distribution
        )
        
    except Exception as e:
        logger.error(f"Error calculating market stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate market stats: {str(e)}")

# Neighborhood information endpoint
@app.get("/neighborhoods", response_model=List[NeighborhoodInfo])
async def get_neighborhoods(city: Optional[str] = None):
    """Get neighborhood information with average prices"""
    try:
        # Load property data - try multiple possible paths
        possible_paths = [
            "../data/raw_properties.csv",
            "data/raw_properties.csv", 
            "../../data/raw_properties.csv"
        ]
        
        df = None
        for data_path in possible_paths:
            if Path(data_path).exists():
                df = pd.read_csv(data_path)
                break
        
        if df is None:
            df = predictor._create_sample_data() if predictor else pd.DataFrame()
        
        if df.empty:
            return []
        
        # Filter by city if provided
        if city:
            df = df[df['city'].str.contains(city, case=False, na=False)]
        
        # Group by neighborhood
        neighborhood_stats = df.groupby(['city', 'neighborhood']).agg({
            'price': ['mean', 'count'],
            'size': 'mean',
            'latitude': 'mean',
            'longitude': 'mean'
        }).round(2).reset_index()
        
        # Flatten column names
        neighborhood_stats.columns = ['city', 'neighborhood', 'avg_price', 'property_count', 
                                    'avg_size', 'latitude', 'longitude']
        
        # Calculate price per sqm
        neighborhood_stats['avg_price_per_sqm'] = (
            neighborhood_stats['avg_price'] / neighborhood_stats['avg_size']
        ).round(2)
        
        # Convert to response format
        neighborhoods = []
        for _, row in neighborhood_stats.iterrows():
            neighborhood = NeighborhoodInfo(
                name=row['neighborhood'],
                city=row['city'],
                avg_price=float(row['avg_price']),
                avg_price_per_sqm=float(row['avg_price_per_sqm']),
                property_count=int(row['property_count']),
                latitude=float(row['latitude']),
                longitude=float(row['longitude'])
            )
            neighborhoods.append(neighborhood)
        
        return neighborhoods
        
    except Exception as e:
        logger.error(f"Error fetching neighborhoods: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch neighborhoods: {str(e)}")

# Model information endpoint
@app.get("/model-info")
async def get_model_info():
    """Get information about the trained model"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        info = {
            "model_name": predictor.best_model_name,
            "model_scores": predictor.model_scores,
            "feature_importance": predictor.feature_importance,
            "training_date": datetime.now().isoformat(),
            "available_cities": ["Sofia", "Plovdiv", "Varna", "Burgas", "Stara Zagora"],
            "supported_features": [
                "parking", "balcony", "elevator", "terrace", "air conditioning",
                "heating", "furnished", "garage"
            ]
        }
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

# Batch prediction endpoint
@app.post("/predict-batch")
async def predict_batch(properties: List[PropertyInput]):
    """Predict prices for multiple properties"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(properties) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 properties per batch")
    
    try:
        predictions = []
        for prop in properties:
            property_data = prop.dict()
            
            if prop.latitude and prop.longitude:
                property_data['coordinates'] = {
                    'lat': prop.latitude,
                    'lng': prop.longitude
                }
            
            prediction = predictor.predict(property_data)
            price_per_sqm = prediction['predicted_price'] / prop.size
            
            pred_result = PricePrediection(
                predicted_price=prediction['predicted_price'],
                confidence_lower=prediction['confidence_lower'],
                confidence_upper=prediction['confidence_upper'],
                price_per_sqm=price_per_sqm,
                model_used=prediction['model_used'],
                prediction_timestamp=datetime.now().isoformat()
            )
            predictions.append(pred_result)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Bulgarian Real Estate Price Predictor API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "/predict",
            "properties": "/properties",
            "market_stats": "/market-stats",
            "neighborhoods": "/neighborhoods",
            "model_info": "/model-info"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 