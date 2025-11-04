from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import joblib
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IRIS Classification API",
    description="MLOps End-to-End Project - IRIS Species Classification with Multiple Models",
    version="1.0.0"
)

# Model paths
MODELS_DIR = Path("models")
AVAILABLE_MODELS = {
    "decision_tree": MODELS_DIR / "decision_tree_model.pkl",
    "logistic_regression": MODELS_DIR / "logistic_regression_model.pkl",
    "random_forest": MODELS_DIR / "random_forest_model.pkl",
    "svm": MODELS_DIR / "support_vector_machine_model.pkl",
    "xgboost": MODELS_DIR / "xgboost_model.pkl"
}

# Load all models at startup
models = {}
for name, path in AVAILABLE_MODELS.items():
    try:
        models[name] = joblib.load(path)
        logger.info(f"Successfully loaded model: {name}")
    except Exception as e:
        logger.error(f"Failed to load model {name}: {str(e)}")

# IRIS species mapping
SPECIES_MAPPING = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}

# Prometheus metrics storage
class MetricsStore:
    def __init__(self):
        self.total_predictions = 0
        self.predictions_by_model = {model: 0 for model in AVAILABLE_MODELS.keys()}
        self.predictions_by_species = {species: 0 for species in SPECIES_MAPPING.values()}
        self.total_errors = 0
        self.request_durations = []
        self.start_time = time.time()
    
    def record_prediction(self, model_name: str, species: str, duration: float):
        self.total_predictions += 1
        self.predictions_by_model[model_name] += 1
        self.predictions_by_species[species] += 1
        self.request_durations.append(duration)
        # Keep only last 1000 durations
        if len(self.request_durations) > 1000:
            self.request_durations = self.request_durations[-1000:]
    
    def record_error(self):
        self.total_errors += 1
    
    def get_uptime(self):
        return time.time() - self.start_time

metrics_store = MetricsStore()

# Pydantic models
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm", ge=0, le=10)
    sepal_width: float = Field(..., description="Sepal width in cm", ge=0, le=10)
    petal_length: float = Field(..., description="Petal length in cm", ge=0, le=10)
    petal_width: float = Field(..., description="Petal width in cm", ge=0, le=10)
    
    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class PredictionRequest(BaseModel):
    features: IrisFeatures
    model_name: Optional[str] = Field(default="random_forest", description="Model to use for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                },
                "model_name": "random_forest"
            }
        }

class PredictionResponse(BaseModel):
    species: str
    species_code: int
    model_used: str
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    timestamp: str

class BatchPredictionRequest(BaseModel):
    features_list: List[IrisFeatures]
    model_name: Optional[str] = Field(default="random_forest", description="Model to use for predictions")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_predictions: int
    model_used: str

# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "IRIS Classification API",
        "version": "1.0.0",
        "available_models": list(AVAILABLE_MODELS.keys()),
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "predict_all_models": "/predict/all-models",
            "health": "/health",
            "models": "/models",
            "metrics": "/metrics"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(models),
        "uptime_seconds": round(metrics_store.get_uptime(), 2)
    }

@app.get("/models", tags=["Models"])
async def get_models():
    """Get information about available models"""
    return {
        "available_models": list(AVAILABLE_MODELS.keys()),
        "models_loaded": list(models.keys()),
        "default_model": "random_forest",
        "species_mapping": SPECIES_MAPPING
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a prediction using the specified model
    """
    start_time = time.time()
    
    try:
        # Validate model name
        if request.model_name not in models:
            metrics_store.record_error()
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_name}' not found. Available models: {list(models.keys())}"
            )
        
        # Get model
        model = models[request.model_name]
        
        # Prepare features
        features = np.array([[
            request.features.sepal_length,
            request.features.sepal_width,
            request.features.petal_length,
            request.features.petal_width
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        species = SPECIES_MAPPING[prediction]
        
        # Get probabilities if available
        probabilities = None
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            probabilities = {
                SPECIES_MAPPING[i]: float(prob) for i, prob in enumerate(proba)
            }
            confidence = float(max(proba))
        
        # Record metrics
        duration = time.time() - start_time
        metrics_store.record_prediction(request.model_name, species, duration)
        
        return PredictionResponse(
            species=species,
            species_code=int(prediction),
            model_used=request.model_name,
            confidence=confidence,
            probabilities=probabilities,
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        metrics_store.record_error()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Make predictions for multiple samples using the specified model
    """
    start_time = time.time()
    
    try:
        # Validate model name
        if request.model_name not in models:
            metrics_store.record_error()
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_name}' not found. Available models: {list(models.keys())}"
            )
        
        # Get model
        model = models[request.model_name]
        
        predictions = []
        
        for features in request.features_list:
            # Prepare features
            feature_array = np.array([[
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width
            ]])
            
            # Make prediction
            prediction = model.predict(feature_array)[0]
            species = SPECIES_MAPPING[prediction]
            
            # Get probabilities if available
            probabilities = None
            confidence = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(feature_array)[0]
                probabilities = {
                    SPECIES_MAPPING[i]: float(prob) for i, prob in enumerate(proba)
                }
                confidence = float(max(proba))
            
            predictions.append(PredictionResponse(
                species=species,
                species_code=int(prediction),
                model_used=request.model_name,
                confidence=confidence,
                probabilities=probabilities,
                timestamp=datetime.now().isoformat()
            ))
            
            # Record metrics
            metrics_store.record_prediction(request.model_name, species, 0)
        
        # Record total duration
        duration = time.time() - start_time
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            model_used=request.model_name
        )
    
    except HTTPException:
        raise
    except Exception as e:
        metrics_store.record_error()
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict/all-models", tags=["Prediction"])
async def predict_all_models(features: IrisFeatures):
    """
    Make predictions using all available models
    """
    start_time = time.time()
    
    try:
        # Prepare features
        feature_array = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        results = {}
        
        for model_name, model in models.items():
            # Make prediction
            prediction = model.predict(feature_array)[0]
            species = SPECIES_MAPPING[prediction]
            
            # Get probabilities if available
            probabilities = None
            confidence = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(feature_array)[0]
                probabilities = {
                    SPECIES_MAPPING[i]: float(prob) for i, prob in enumerate(proba)
                }
                confidence = float(max(proba))
            
            results[model_name] = {
                "species": species,
                "species_code": int(prediction),
                "confidence": confidence,
                "probabilities": probabilities
            }
            
            # Record metrics
            metrics_store.record_prediction(model_name, species, 0)
        
        duration = time.time() - start_time
        
        return {
            "predictions": results,
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": round(duration, 4)
        }
    
    except Exception as e:
        metrics_store.record_error()
        logger.error(f"All models prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics", response_class=PlainTextResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Prometheus-compatible metrics endpoint
    """
    avg_duration = sum(metrics_store.request_durations) / len(metrics_store.request_durations) if metrics_store.request_durations else 0
    
    metrics = []
    
    # Help and type declarations
    metrics.append("# HELP iris_predictions_total Total number of predictions made")
    metrics.append("# TYPE iris_predictions_total counter")
    metrics.append(f"iris_predictions_total {metrics_store.total_predictions}")
    metrics.append("")
    
    metrics.append("# HELP iris_predictions_by_model_total Total predictions by model")
    metrics.append("# TYPE iris_predictions_by_model_total counter")
    for model, count in metrics_store.predictions_by_model.items():
        metrics.append(f'iris_predictions_by_model_total{{model="{model}"}} {count}')
    metrics.append("")
    
    metrics.append("# HELP iris_predictions_by_species_total Total predictions by species")
    metrics.append("# TYPE iris_predictions_by_species_total counter")
    for species, count in metrics_store.predictions_by_species.items():
        metrics.append(f'iris_predictions_by_species_total{{species="{species}"}} {count}')
    metrics.append("")
    
    metrics.append("# HELP iris_prediction_errors_total Total number of prediction errors")
    metrics.append("# TYPE iris_prediction_errors_total counter")
    metrics.append(f"iris_prediction_errors_total {metrics_store.total_errors}")
    metrics.append("")
    
    metrics.append("# HELP iris_prediction_duration_seconds Average prediction duration")
    metrics.append("# TYPE iris_prediction_duration_seconds gauge")
    metrics.append(f"iris_prediction_duration_seconds {avg_duration:.6f}")
    metrics.append("")
    
    metrics.append("# HELP iris_api_uptime_seconds API uptime in seconds")
    metrics.append("# TYPE iris_api_uptime_seconds counter")
    metrics.append(f"iris_api_uptime_seconds {metrics_store.get_uptime():.2f}")
    metrics.append("")
    
    metrics.append("# HELP iris_models_loaded Number of models loaded")
    metrics.append("# TYPE iris_models_loaded gauge")
    metrics.append(f"iris_models_loaded {len(models)}")
    metrics.append("")
    
    return "\n".join(metrics)

@app.get("/metrics/summary", tags=["Monitoring"])
async def get_metrics_summary():
    """
    Get metrics summary in JSON format
    """
    avg_duration = sum(metrics_store.request_durations) / len(metrics_store.request_durations) if metrics_store.request_durations else 0
    
    return {
        "total_predictions": metrics_store.total_predictions,
        "predictions_by_model": metrics_store.predictions_by_model,
        "predictions_by_species": metrics_store.predictions_by_species,
        "total_errors": metrics_store.total_errors,
        "average_prediction_duration_seconds": round(avg_duration, 6),
        "uptime_seconds": round(metrics_store.get_uptime(), 2),
        "models_loaded": len(models),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
