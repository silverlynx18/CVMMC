"""FastAPI REST API for pedestrian counting and analysis."""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from .processing_engine import ProcessingEngine
from .models import Camera, PedestrianCount, ServiceLevelAnalysis, Alert
from .fruin_analysis import FruinAnalyzer
from .config import settings

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Pedestrian Counting and Service Level Analysis API",
    description="API for real-time pedestrian counting and Fruin's Level of Service analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processing engine instance
processing_engine = None


# Pydantic models for API requests/responses
class CameraCreate(BaseModel):
    name: str
    location: str
    ip_address: str
    port: int = 554
    username: str
    password: str
    calibration_data: Optional[Dict[str, Any]] = None


class CameraResponse(BaseModel):
    id: int
    name: str
    location: str
    ip_address: str
    port: int
    is_active: bool
    status: str
    created_at: datetime


class PedestrianCountResponse(BaseModel):
    camera_id: int
    timestamp: datetime
    ingress_count: int
    egress_count: int
    net_count: int
    current_pedestrians: int
    density_per_sqm: float
    fruin_los: str


class LOSAnalysisResponse(BaseModel):
    camera_id: int
    timestamp: datetime
    avg_density: float
    peak_density: float
    avg_los: str
    peak_los: str
    total_ingress: int
    total_egress: int
    peak_hour_factor: float


class AlertResponse(BaseModel):
    id: int
    camera_id: int
    timestamp: datetime
    alert_type: str
    severity: str
    message: str
    is_resolved: bool


class ZoneConfiguration(BaseModel):
    zones: Dict[str, Dict[str, Any]]
    ingress_zones: List[str]
    egress_zones: List[str]


# Dependency to get processing engine
async def get_processing_engine():
    global processing_engine
    if processing_engine is None:
        processing_engine = ProcessingEngine()
        await processing_engine.initialize()
    return processing_engine


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the processing engine on startup."""
    global processing_engine
    processing_engine = ProcessingEngine()
    await processing_engine.initialize()
    logger.info("API started successfully")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Stop the processing engine on shutdown."""
    global processing_engine
    if processing_engine:
        await processing_engine.stop_processing()
    logger.info("API shutdown complete")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# Camera management endpoints
@app.get("/cameras", response_model=List[CameraResponse])
async def get_cameras(engine: ProcessingEngine = Depends(get_processing_engine)):
    """Get all cameras."""
    try:
        camera_status = engine.get_camera_status()
        # In real implementation, this would query the database
        cameras = []
        for camera_id, status in camera_status.items():
            cameras.append(CameraResponse(
                id=camera_id,
                name=f"Camera_{camera_id:02d}",
                location=f"Station_Zone_{camera_id}",
                ip_address=f"192.168.1.{100 + camera_id}",
                port=554,
                is_active=status == "online",
                status=status,
                created_at=datetime.utcnow()
            ))
        return cameras
    except Exception as e:
        logger.error(f"Error getting cameras: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cameras")


@app.post("/cameras", response_model=CameraResponse)
async def create_camera(camera_data: CameraCreate, 
                       engine: ProcessingEngine = Depends(get_processing_engine)):
    """Create a new camera."""
    try:
        # In real implementation, this would create camera in database
        # and add to processing engine
        camera_id = len(engine.camera_manager.cameras) + 1
        
        return CameraResponse(
            id=camera_id,
            name=camera_data.name,
            location=camera_data.location,
            ip_address=camera_data.ip_address,
            port=camera_data.port,
            is_active=True,
            status="offline",
            created_at=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error creating camera: {e}")
        raise HTTPException(status_code=500, detail="Failed to create camera")


@app.get("/cameras/{camera_id}/status")
async def get_camera_status(camera_id: int, 
                           engine: ProcessingEngine = Depends(get_processing_engine)):
    """Get status of a specific camera."""
    try:
        status = engine.get_camera_status().get(camera_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        return {
            "camera_id": camera_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting camera status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get camera status")


# Pedestrian counting endpoints
@app.get("/counts", response_model=Dict[str, PedestrianCountResponse])
async def get_latest_counts(camera_id: Optional[int] = None,
                           engine: ProcessingEngine = Depends(get_processing_engine)):
    """Get latest pedestrian counts."""
    try:
        counts = engine.get_latest_counts(camera_id)
        
        if camera_id is not None:
            if camera_id not in counts:
                raise HTTPException(status_code=404, detail="Camera not found")
            
            count_data = counts[camera_id]
            return {
                str(camera_id): PedestrianCountResponse(
                    camera_id=camera_id,
                    timestamp=datetime.utcnow(),
                    ingress_count=count_data.get('ingress_count', 0),
                    egress_count=count_data.get('egress_count', 0),
                    net_count=count_data.get('net_count', 0),
                    current_pedestrians=count_data.get('current_pedestrians', 0),
                    density_per_sqm=0.0,  # Would be calculated
                    fruin_los="A"  # Would be calculated
                )
            }
        
        # Return counts for all cameras
        result = {}
        for cid, count_data in counts.items():
            result[str(cid)] = PedestrianCountResponse(
                camera_id=cid,
                timestamp=datetime.utcnow(),
                ingress_count=count_data.get('ingress_count', 0),
                egress_count=count_data.get('egress_count', 0),
                net_count=count_data.get('net_count', 0),
                current_pedestrians=count_data.get('current_pedestrians', 0),
                density_per_sqm=0.0,
                fruin_los="A"
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting counts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get counts")


# Level of Service endpoints
@app.get("/los", response_model=Dict[str, LOSAnalysisResponse])
async def get_latest_los(camera_id: Optional[int] = None,
                        engine: ProcessingEngine = Depends(get_processing_engine)):
    """Get latest Level of Service analysis."""
    try:
        los_data = engine.get_latest_los(camera_id)
        
        if camera_id is not None:
            if str(camera_id) not in los_data:
                raise HTTPException(status_code=404, detail="Camera not found")
            
            data = los_data[str(camera_id)]
            return {
                str(camera_id): LOSAnalysisResponse(
                    camera_id=data['camera_id'],
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    avg_density=data['density'],
                    peak_density=data['density'] * 1.5,  # Sample calculation
                    avg_los=data['los_level'],
                    peak_los=chr(ord(data['los_level']) + 1) if data['los_level'] != 'F' else 'F',
                    total_ingress=0,  # Would be calculated
                    total_egress=0,   # Would be calculated
                    peak_hour_factor=1.0
                )
            }
        
        # Return LOS for all cameras
        result = {}
        for cid, data in los_data.items():
            result[cid] = LOSAnalysisResponse(
                camera_id=data['camera_id'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                avg_density=data['density'],
                peak_density=data['density'] * 1.5,
                avg_los=data['los_level'],
                peak_los=chr(ord(data['los_level']) + 1) if data['los_level'] != 'F' else 'F',
                total_ingress=0,
                total_egress=0,
                peak_hour_factor=1.0
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting LOS: {e}")
        raise HTTPException(status_code=500, detail="Failed to get LOS data")


@app.get("/los/summary")
async def get_los_summary():
    """Get Fruin's Level of Service summary."""
    try:
        analyzer = FruinAnalyzer()
        return analyzer.get_los_summary()
    except Exception as e:
        logger.error(f"Error getting LOS summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get LOS summary")


# Historical analysis endpoints
@app.get("/analysis/historical/{camera_id}")
async def get_historical_analysis(camera_id: int,
                                 start_time: Optional[datetime] = None,
                                 end_time: Optional[datetime] = None,
                                 engine: ProcessingEngine = Depends(get_processing_engine)):
    """Get historical analysis for a camera."""
    try:
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.utcnow()
        
        analysis = engine.get_historical_analysis(camera_id, start_time, end_time)
        return analysis
    except Exception as e:
        logger.error(f"Error getting historical analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to get historical analysis")


# Zone configuration endpoints
@app.post("/cameras/{camera_id}/zones")
async def configure_zones(camera_id: int,
                         zone_config: ZoneConfiguration,
                         engine: ProcessingEngine = Depends(get_processing_engine)):
    """Configure detection zones for a camera."""
    try:
        if camera_id not in engine.camera_manager.cameras:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        # Set zones for camera
        engine.camera_manager.set_camera_zones(
            camera_id,
            zone_config.zones,
            zone_config.ingress_zones,
            zone_config.egress_zones
        )
        
        # Set zones for tracker
        if camera_id in engine.trackers:
            engine.trackers[camera_id].set_zones(
                zone_config.zones,
                zone_config.ingress_zones,
                zone_config.egress_zones
            )
        
        return {"message": f"Zones configured for camera {camera_id}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring zones: {e}")
        raise HTTPException(status_code=500, detail="Failed to configure zones")


# Control endpoints
@app.post("/start")
async def start_processing(engine: ProcessingEngine = Depends(get_processing_engine)):
    """Start the processing engine."""
    try:
        await engine.start_processing()
        return {"message": "Processing started successfully"}
    except Exception as e:
        logger.error(f"Error starting processing: {e}")
        raise HTTPException(status_code=500, detail="Failed to start processing")


@app.post("/stop")
async def stop_processing(engine: ProcessingEngine = Depends(get_processing_engine)):
    """Stop the processing engine."""
    try:
        await engine.stop_processing()
        return {"message": "Processing stopped successfully"}
    except Exception as e:
        logger.error(f"Error stopping processing: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop processing")


@app.get("/status")
async def get_engine_status(engine: ProcessingEngine = Depends(get_processing_engine)):
    """Get processing engine status."""
    try:
        camera_status = engine.get_camera_status()
        online_cameras = sum(1 for status in camera_status.values() if status == "online")
        
        return {
            "is_running": engine.is_running,
            "total_cameras": len(camera_status),
            "online_cameras": online_cameras,
            "camera_status": camera_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting engine status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get engine status")


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)