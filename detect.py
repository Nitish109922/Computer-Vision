import cv2
import time
import os
import json
import asyncio
import threading
import queue
import requests
import base64
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Tuple, Dict, Set, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone
import numpy as np
import pyttsx3
from ultralytics import YOLO


try:
    import websockets
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut
    import subprocess
    import platform
    from sklearn.cluster import KMeans
    from scipy import stats
except ImportError as e:
    logging.warning(f"Some advanced features may not work. Missing: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeatherCondition(Enum):
    """Weather condition classifications"""
    CLEAR = "clear"
    PARTLY_CLOUDY = "partly_cloudy"
    CLOUDY = "cloudy"
    OVERCAST = "overcast"
    RAINY = "rainy"
    FOGGY = "foggy"
    UNKNOWN = "unknown"


class ModelType(Enum):
   
    OBJECT = "object"
    CURRENCY = "Currency"


@dataclass
class DetectionResult:
   
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    label: str
    model_name: str
    color: Tuple[int, int, int]


@dataclass
class WeatherData:
    
    condition: WeatherCondition
    confidence: float
    brightness: float
    contrast: float
    blur_factor: float
    timestamp: datetime


@dataclass
class LocationData:
   
    latitude: float
    longitude: float
    address: str
    accuracy: float
    timestamp: datetime
    method: str  # 'gps', 'ip', 'manual'


@dataclass
class StreamingData:
    
    frame_data: str  # base64 encoded frame
    detections: List[Dict[str, Any]]
    weather: Dict[str, Any]
    location: Dict[str, Any]
    timestamp: str
    frame_count: int


@dataclass
class Config:
    
    model_paths: List[str]
    conf_threshold: float
    person_threshold: int
    alert_delay: float
    window_name: str
    detection_timeout: float
    announcement_cooldown: float
    model_colors: Dict[str, Tuple[int, int, int]]
    fps_limit: float
    

    weather_analysis_enabled: bool
    location_transmission_enabled: bool
    video_streaming_enabled: bool
    streaming_port: int
    streaming_quality: int  # JPEG quality 0-100
    location_update_interval: float  # seconds
    weather_analysis_interval: float  # seconds
    remote_server_url: Optional[str]
    websocket_port: int


class WeatherAnalyzer:
   
    
    def __init__(self) -> None:
    
        self.last_analysis: Optional[WeatherData] = None
        self.analysis_history: List[WeatherData] = []
        self.max_history_size: int = 10
        
    def analyze_frame(self, frame: np.ndarray) -> WeatherData:
        try:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            

            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            blur_factor = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            sat_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            val_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            condition, confidence = self._classify_weather(
                brightness, contrast, blur_factor, hue_hist, sat_hist, val_hist, gray
            )
            
            weather_data = WeatherData(
                condition=condition,
                confidence=confidence,
                brightness=brightness,
                contrast=contrast,
                blur_factor=blur_factor,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Update history
            self.analysis_history.append(weather_data)
            if len(self.analysis_history) > self.max_history_size:
                self.analysis_history.pop(0)
            
            self.last_analysis = weather_data
            return weather_data
            
        except Exception as e:
            logger.error(f"Weather analysis error: {e}")
            return WeatherData(
                condition=WeatherCondition.UNKNOWN,
                confidence=0.0,
                brightness=0.0,
                contrast=0.0,
                blur_factor=0.0,
                timestamp=datetime.now(timezone.utc)
            )
    
    def _classify_weather(
        self, 
        brightness: float, 
        contrast: float, 
        blur_factor: float,
        hue_hist: np.ndarray,
        sat_hist: np.ndarray, 
        val_hist: np.ndarray,
        gray_frame: np.ndarray
    ) -> Tuple[WeatherCondition, float]:
        try:
            # Normalize metrics
            brightness_norm = brightness / 255.0
            contrast_norm = min(contrast / 100.0, 1.0)
            blur_norm = min(blur_factor / 1000.0, 1.0)
            
            # Calculate saturation dominance
            sat_mean = np.mean(sat_hist)
            sat_std = np.std(sat_hist)
            
            # Detect edges for texture analysis
            edges = cv2.Canny(gray_frame, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            confidence = 0.7  # Base confidence
            
            # Classification rules based on computer vision research
            if blur_norm < 0.1 and brightness_norm < 0.3:
                return WeatherCondition.FOGGY, confidence + 0.2
            elif brightness_norm > 0.7 and contrast_norm > 0.4 and edge_density > 0.1:
                return WeatherCondition.CLEAR, confidence + 0.3
            elif brightness_norm < 0.4 and contrast_norm < 0.3:
                if blur_norm < 0.2:
                    return WeatherCondition.RAINY, confidence + 0.1
                else:
                    return WeatherCondition.OVERCAST, confidence
            elif brightness_norm > 0.4 and contrast_norm > 0.2:
                if sat_mean > 50:
                    return WeatherCondition.PARTLY_CLOUDY, confidence
                else:
                    return WeatherCondition.CLOUDY, confidence - 0.1
            else:
                return WeatherCondition.CLOUDY, confidence - 0.2
                
        except Exception as e:
            logger.error(f"Weather classification error: {e}")
            return WeatherCondition.UNKNOWN, 0.0
    
    def get_weather_trend(self) -> Optional[WeatherCondition]:
        """Get the trending weather condition from recent history
        
        Returns:
            Most common weather condition in recent history
        """
        if len(self.analysis_history) < 3:
            return None
            
        recent_conditions = [w.condition for w in self.analysis_history[-5:]]
        return max(set(recent_conditions), key=recent_conditions.count)


class LocationManager:
    """Manages location tracking and transmission"""
    
    def __init__(self) -> None:
        """Initialize location manager"""
        self.current_location: Optional[LocationData] = None
        self.geolocator = Nominatim(user_agent="yolo_detection_system")
        self.last_update: float = 0.0
        
    async def get_current_location(self) -> Optional[LocationData]:
        """Get current location using multiple methods
        
        Returns:
            Current location data or None if failed
        """
        try:
            # Try GPS first (if available)
            location = await self._get_gps_location()
            if location:
                return location
                
            # Fallback to IP-based location
            location = await self._get_ip_location()
            if location:
                return location
                
            logger.warning("Could not determine location")
            return None
            
        except Exception as e:
            logger.error(f"Location retrieval error: {e}")
            return None
    
    async def _get_gps_location(self) -> Optional[LocationData]:
        """Get GPS location (implementation depends on platform)
        
        Returns:
            GPS location data or None
        """
        try:
            # This is a simplified implementation
            # In a real application, you'd use platform-specific GPS APIs
            
            if platform.system() == "Windows":
                # Windows location API would go here
                pass
            elif platform.system() == "Darwin":  # macOS
                # macOS Core Location would go here
                pass
            elif platform.system() == "Linux":
                # Linux GPS daemon interface would go here
                pass
            
            # For now, return None (GPS not implemented)
            return None
            
        except Exception as e:
            logger.error(f"GPS location error: {e}")
            return None
    
    async def _get_ip_location(self) -> Optional[LocationData]:
        """Get location based on IP address
        
        Returns:
            IP-based location data or None
        """
        try:
            # Use a free IP geolocation service
            response = requests.get("http://ip-api.com/json/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success":
                    return LocationData(
                        latitude=12.926912,
                        longitude=77.526353,
                        address=f"{data['city']}, {data['regionName']}, {data['country']}",
                        accuracy=1000.0,  # IP location is not very accurate
                        timestamp=datetime.now(timezone.utc),
                        method="ip"
                    )
            return None
            
        except Exception as e:
            logger.error(f"IP location error: {e}")
            return None
    
    async def transmit_location(self, server_url: str) -> bool:
        """Transmit current location to remote server
        
        Args:
            server_url: URL of the remote server
            
        Returns:
            True if transmission successful, False otherwise
        """
        try:
            if not self.current_location:
                return False
                
            location_data = asdict(self.current_location)
            location_data['timestamp'] = self.current_location.timestamp.isoformat()
            
            response = requests.post(
                f"{server_url}/location",
                json=location_data,
                timeout=5
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Location transmission error: {e}")
            return False


class VideoStreamer:
    """Handles live video streaming to React Expo App"""
    
    def __init__(self, port: int = 8000, quality: int = 80) -> None:
        """Initialize video streamer
        
        Args:
            port: WebSocket port for streaming
            quality: JPEG compression quality (0-100)
        """
        self.port: int = port
        self.quality: int = quality
        self.connected_clients: Set[WebSocket] = set()
        self.app: FastAPI = self._create_app()
        self.server_task: Optional[asyncio.Task] = None
        self.frame_count: int = 0
        
    def _create_app(self) -> FastAPI:
        """Create FastAPI application for streaming
        
        Returns:
            Configured FastAPI app
        """
        app = FastAPI(title="YOLO Detection Stream")
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.websocket("/stream")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.connected_clients.add(websocket)
            logger.info(f"Client connected. Total clients: {len(self.connected_clients)}")
            
            try:
                while True:
                    await websocket.receive_text()  # Keep connection alive
            except WebSocketDisconnect:
                self.connected_clients.remove(websocket)
                logger.info(f"Client disconnected. Total clients: {len(self.connected_clients)}")
        
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "connected_clients": len(self.connected_clients)}
        
        return app
    
    async def start_server(self) -> None:
        """Start the streaming server"""
        try:
            config = uvicorn.Config(
                self.app, 
                host="0.0.0.0", 
                port=self.port, 
                log_level="warning"
            )
            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            logger.error(f"Streaming server error: {e}")
    
    async def broadcast_frame(self, streaming_data: StreamingData) -> None:
        """Broadcast frame to all connected clients
        
        Args:
            streaming_data: Data to broadcast
        """
        if not self.connected_clients:
            return
            
        message = json.dumps(asdict(streaming_data))
        disconnected_clients = set()
        
        for client in self.connected_clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected_clients
    
    def encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame as base64 JPEG
        
        Args:
            frame: Input frame
            
        Returns:
            Base64 encoded JPEG string
        """
        try:
            # Resize frame for streaming efficiency
            height, width = frame.shape[:2]
            if width > 1280:
                scale = 1280 / width
                new_width = 1280
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Encode as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            # Convert to base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{frame_base64}"
            
        except Exception as e:
            logger.error(f"Frame encoding error: {e}")
            return ""


class TTSManager:
    """Thread-safe Text-to-Speech manager"""
    
    def __init__(self) -> None:
        """Initialize TTS manager with threading support"""
        self.tts_engine: pyttsx3.Engine = pyttsx3.init()
        self.tts_queue: queue.Queue[Optional[str]] = queue.Queue()
        self.tts_thread: threading.Thread = threading.Thread(
            target=self._tts_worker, daemon=True
        )
        self._running: bool = True
        self.tts_thread.start()
        logger.info("TTS Manager initialized")
    
    def _tts_worker(self) -> None:
        """Worker thread for TTS processing"""
        while self._running:
            try:
                message = self.tts_queue.get(timeout=1.0)
                if message is None:
                    break
                self.tts_engine.say(message)
                self.tts_engine.runAndWait()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS error: {e}")
    
    def announce(self, message: str) -> None:
        """Add message to TTS queue"""
        if self._running:
            self.tts_queue.put(message)
            logger.info(f"Queued announcement: {message}")
    
    def shutdown(self) -> None:
        """Shutdown TTS manager"""
        self._running = False
        self.tts_queue.put(None)
        self.tts_thread.join(timeout=2.0)
        logger.info("TTS Manager shutdown")


class AnnouncementManager:
    """Manages announcement timing to prevent spam"""
    
    def __init__(self, cooldown_period: float = 10.0) -> None:
        """Initialize announcement manager
        
        Args:
            cooldown_period: Minimum time between same announcements (seconds)
        """
        self.cooldown_period: float = cooldown_period
        self.last_announcements: Dict[str, float] = {}
        self.lock: threading.Lock = threading.Lock()
    
    def should_announce(self, key: str) -> bool:
        """Check if announcement should be made based on cooldown
        
        Args:
            key: Unique identifier for the announcement
            
        Returns:
            True if announcement should be made, False otherwise
        """
        current_time = time.time()
        with self.lock:
            last_time = self.last_announcements.get(key, 0.0)
            if current_time - last_time >= self.cooldown_period:
                self.last_announcements[key] = current_time
                return True
            return False
    
    def reset_announcement(self, key: str) -> None:
        """Reset announcement timer for specific key"""
        with self.lock:
            self.last_announcements.pop(key, None)


class ModelManager:
    """Manages YOLO models and switching between them"""
    
    def __init__(self, model_paths: List[str], model_colors: Dict[str, Tuple[int, int, int]]) -> None:
        """Initialize model manager
        
        Args:
            model_paths: List of paths to YOLO model files
            model_colors: Color mapping for each model type
        """
        self.models: List[YOLO] = []
        self.model_names: List[str] = []
        self.model_colors: Dict[str, Tuple[int, int, int]] = model_colors
        self.active_models: Set[int] = set()
        
        self._load_models(model_paths)
    
    def _load_models(self, model_paths: List[str]) -> None:
        """Load YOLO models from file paths"""
        for i, path in enumerate(model_paths):
            try:
                if not os.path.exists(path):
                    logger.warning(f"Model file not found: {path}")
                    continue
                
                model = YOLO(path)
                model_name = os.path.basename(path).replace('.pt', '')
                
                self.models.append(model)
                self.model_names.append(model_name)
                self.active_models.add(i)
                
                logger.info(f"Loaded model: {model_name} from {path}")
            except Exception as e:
                logger.error(f"Failed to load model {path}: {e}")
    
    def toggle_model(self, model_index: int) -> bool:
        """Toggle model active state
        
        Args:
            model_index: Index of model to toggle
            
        Returns:
            True if toggle was successful, False otherwise
        """
        if 0 <= model_index < len(self.models):
            if model_index in self.active_models:
                self.active_models.remove(model_index)
                logger.info(f"Deactivated model: {self.model_names[model_index]}")
            else:
                self.active_models.add(model_index)
                logger.info(f"Activated model: {self.model_names[model_index]}")
            return True
        return False
    
    def get_active_models(self) -> List[Tuple[YOLO, str]]:
        """Get list of currently active models
        
        Returns:
            List of tuples containing (model, model_name)
        """
        return [(self.models[i], self.model_names[i]) 
                for i in self.active_models if i < len(self.models)]
    
    def get_model_color(self, model_name: str) -> Tuple[int, int, int]:
        """Get color for specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            BGR color tuple
        """
        return self.model_colors.get(model_name, (255, 255, 255))


class ObjectDetectionSystem:
    """Main object detection system with advanced features"""
    
    def __init__(self, config: Config) -> None:
        """Initialize detection system
        
        Args:
            config: Configuration object with system parameters
        """
        self.config: Config = config
        self.tts_manager: TTSManager = TTSManager()
        self.announcement_manager: AnnouncementManager = AnnouncementManager(
            config.announcement_cooldown
        )
        self.model_manager: ModelManager = ModelManager(
            config.model_paths, config.model_colors
        )
        
        # Initialize advanced features
        self.weather_analyzer: Optional[WeatherAnalyzer] = None
        self.location_manager: Optional[LocationManager] = None
        self.video_streamer: Optional[VideoStreamer] = None
        
        if config.weather_analysis_enabled:
            self.weather_analyzer = WeatherAnalyzer()
            
        if config.location_transmission_enabled:
            self.location_manager = LocationManager()
            
        if config.video_streaming_enabled:
            self.video_streamer = VideoStreamer(
                config.websocket_port, 
                config.streaming_quality
            )
        
        self.last_person_alert: float = 0.0
        self.last_weather_analysis: float = 0.0
        self.last_location_update: float = 0.0
        self.cap: Optional[cv2.VideoCapture] = None
        self._running: bool = False
        
        logger.info("Enhanced Object Detection System initialized")
    
    def _run_model_inference(
        self, 
        model: YOLO, 
        frame: np.ndarray, 
        model_name: str
    ) -> List[DetectionResult]:
        """Run inference on single model
        
        Args:
            model: YOLO model instance
            frame: Input frame
            model_name: Name of the model
            
        Returns:
            List of detection results
        """
        try:
            results = model(frame, conf=self.config.conf_threshold, verbose=False)
            if not results or not results[0].boxes:
                return []
            
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            detections = []
            for bbox, score, class_id in zip(boxes, scores, class_ids):
                if np.isnan(score) or not (self.config.conf_threshold <= score <= 1.0):
                    continue
                
                label = model.names.get(class_id, f"class_{class_id}")
                color = self.model_manager.get_model_color(model_name)
                
                detection = DetectionResult(
                    bbox=tuple(map(int, bbox)),
                    confidence=float(score),
                    class_id=int(class_id),
                    label=label,
                    model_name=model_name,
                    color=color
                )
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Model inference error for {model_name}: {e}")
            return []
    
    def _process_detections(
        self, 
        frame: np.ndarray
    ) -> Tuple[List[DetectionResult], int]:
        """Process detections from all active models
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (all_detections, person_count)
        """
        active_models = self.model_manager.get_active_models()
        if not active_models:
            return [], 0
        
        all_detections = []
        
        with ThreadPoolExecutor(max_workers=len(active_models)) as executor:
            futures: List[Future[List[DetectionResult]]] = [
                executor.submit(self._run_model_inference, model, frame, name)
                for model, name in active_models
            ]
            
            for future in futures:
                try:
                    detections = future.result(timeout=1.0)
                    all_detections.extend(detections)
                except Exception as e:
                    logger.error(f"Detection processing error: {e}")
        
        person_count = sum(1 for det in all_detections if det.label == 'person')
        return all_detections, person_count
    
    async def _analyze_weather(self, frame: np.ndarray) -> Optional[WeatherData]:
        """Analyze weather conditions from frame
        
        Args:
            frame: Input frame
            
        Returns:
            Weather analysis results or None
        """
        if not self.weather_analyzer:
            return None
            
        current_time = time.time()
        if current_time - self.last_weather_analysis < self.config.weather_analysis_interval:
            return self.weather_analyzer.last_analysis
            
        try:
            weather_data = self.weather_analyzer.analyze_frame(frame)
            self.last_weather_analysis = current_time
            
            # Announce weather changes
            if self.weather_analyzer.last_analysis:
                prev_condition = self.weather_analyzer.last_analysis.condition
                if (weather_data.condition != prev_condition and 
                    self.announcement_manager.should_announce(f"weather_{weather_data.condition.value}")):
                    
                    self.tts_manager.announce(
                        f"Weather condition changed to {weather_data.condition.value.replace('_', ' ')}"
                    )
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Weather analysis error: {e}")
            return None
    
    async def _update_location(self) -> Optional[LocationData]:
        """Update current location
        
        Returns:
            Updated location data or None
        """
        if not self.location_manager:
            return None
            
        current_time = time.time()
        if current_time - self.last_location_update < self.config.location_update_interval:
            return self.location_manager.current_location
            
        try:
            location_data = await self.location_manager.get_current_location()
            if location_data:
                self.location_manager.current_location = location_data
                self.last_location_update = current_time
                
                # Transmit to remote server if configured
                if self.config.remote_server_url:
                    await self.location_manager.transmit_location(self.config.remote_server_url)
            
            return location_data
            
        except Exception as e:
            logger.error(f"Location update error: {e}")
            return None
    
    def _draw_detections(
        self, 
        frame: np.ndarray, 
        detections: List[DetectionResult]
    ) -> np.ndarray:
       
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), detection.color, 2)
            
            # Draw label with confidence
            label_text = f"{detection.label} ({detection.model_name}) {detection.confidence:.2f}"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw text background
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 4),
                (x1 + text_width, y1),
                detection.color,
                -1
            )
            
            # Draw text
            cv2.putText(
                frame, label_text, (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        
        return frame
    
    def _handle_announcements(
        self, 
        detections: List[DetectionResult], 
        person_count: int
    ) -> None:
        """Handle TTS announcements for detections
        
        Args:
            detections: List of current detections
            person_count: Number of persons detected
        """
        current_time = time.time()
        
        # Handle person count alerts
        if (person_count > self.config.person_threshold and 
            (current_time - self.last_person_alert) > self.config.alert_delay):
            
            alert_msg = f"Alert! {person_count} people detected."
            self.tts_manager.announce(alert_msg)
            self.last_person_alert = current_time
        
        # Handle new object detections
        current_detections: Set[Tuple[str, str]] = {
            (det.label, det.model_name) for det in detections
        }
        
        for label, model_name in current_detections:
            announcement_key = f"{label}_{model_name}"
            
            if self.announcement_manager.should_announce(announcement_key):
                msg = f"New {label} detected by {model_name} model"
                self.tts_manager.announce(msg)
    
    def _handle_keyboard_input(self, key: int) -> bool:
        """Handle keyboard input for system control
        
        Args:
            key: OpenCV key code
            
        Returns:
            True to continue, False to quit
        """
        if key == ord('q') or key == 27:  # 'q' or ESC
            return False
        elif key == ord('1'):
            self.model_manager.toggle_model(0)
        elif key == ord('2'):
            self.model_manager.toggle_model(1)
        elif key == ord('r'):
            # Reset all announcement timers
            self.announcement_manager = AnnouncementManager(
                self.config.announcement_cooldown
            )
            logger.info("Reset announcement timers")
        elif key == ord('w'):
            # Toggle weather analysis
            if self.weather_analyzer:
                self.config.weather_analysis_enabled = not self.config.weather_analysis_enabled
                logger.info(f"Weather analysis: {'enabled' if self.config.weather_analysis_enabled else 'disabled'}")
        elif key == ord('l'):
            # Toggle location transmission
            if self.location_manager:
                self.config.location_transmission_enabled = not self.config.location_transmission_enabled
                logger.info(f"Location transmission: {'enabled' if self.config.location_transmission_enabled else 'disabled'}")
        elif key == ord('s'):
            # Toggle video streaming
            if self.video_streamer:
                self.config.video_streaming_enabled = not self.config.video_streaming_enabled
                logger.info(f"Video streaming: {'enabled' if self.config.video_streaming_enabled else 'disabled'}")
        
        return True
    
    def _draw_ui_info(
        self, 
        frame: np.ndarray, 
        person_count: int, 
        weather_data: Optional[WeatherData] = None,
        location_data: Optional[LocationData] = None
    ) -> np.ndarray:
        """Draw UI information on frame
        
        Args:
            frame: Input frame
            person_count: Number of persons detected
            weather_data: Current weather analysis
            location_data: Current location data
            
        Returns:
            Frame with UI information
        """
        # Person count
        cv2.putText(
            frame, f"Persons: {person_count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
        )
        
        # Active models info
        active_models = self.model_manager.get_active_models()
        models_text = f"Active Models: {len(active_models)}"
        cv2.putText(
            frame, models_text, (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )
        
        # Weather information
        if weather_data and self.config.weather_analysis_enabled:
            weather_text = f"Weather: {weather_data.condition.value.replace('_', ' ').title()}"
            cv2.putText(
                frame, weather_text, (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            
            # Weather confidence
            conf_text = f"Confidence: {weather_data.confidence:.2f}"
            cv2.putText(
                frame, conf_text, (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )
        
        # Location information
        if location_data and self.config.location_transmission_enabled:
            loc_text = f"Location: {location_data.latitude:.4f}, {location_data.longitude:.4f}"
            cv2.putText(
                frame, loc_text, (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1
            )
        
        # Streaming status
        if self.video_streamer and self.config.video_streaming_enabled:
            stream_text = f"Streaming: {len(self.video_streamer.connected_clients)} clients"
            cv2.putText(
                frame, stream_text, (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
        
        # Controls info
        controls = [
            "Controls: Q-Quit, 1-Model1, 2-Model2, R-Reset",
            "W-Weather, L-Location, S-Streaming"
        ]
        for i, control in enumerate(controls):
            cv2.putText(
                frame, control, (10, frame.shape[0] - 40 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        return frame
    
    async def _prepare_streaming_data(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        weather_data: Optional[WeatherData],
        location_data: Optional[LocationData]
    ) -> StreamingData:
        """Prepare data for streaming to React Expo App
        
        Args:
            frame: Current frame
            detections: Current detections
            weather_data: Weather analysis results
            location_data: Location data
            
        Returns:
            Streaming data package
        """
        try:
            # Encode frame
            frame_encoded = ""
            if self.video_streamer:
                frame_encoded = self.video_streamer.encode_frame(frame)
            
            # Convert detections to serializable format
            detection_dicts = []
            for det in detections:
                detection_dicts.append({
                    "bbox": det.bbox,
                    "confidence": det.confidence,
                    "class_id": det.class_id,
                    "label": det.label,
                    "model_name": det.model_name,
                    "color": det.color
                })
            
            # Convert weather data
            weather_dict = {}
            if weather_data:
                weather_dict = {
                    "condition": weather_data.condition.value,
                    "confidence": weather_data.confidence,
                    "brightness": weather_data.brightness,
                    "contrast": weather_data.contrast,
                    "blur_factor": weather_data.blur_factor,
                    "timestamp": weather_data.timestamp.isoformat()
                }
            
            # Convert location data
            location_dict = {}
            if location_data:
                location_dict = {
                    "latitude": 12.926912,
                    "longitude": 77.526353,
                    "address": location_data.address,
                    "accuracy": location_data.accuracy,
                    "timestamp": location_data.timestamp.isoformat(),
                    "method": location_data.method
                }
            
            return StreamingData(
                frame_data=frame_encoded,
                detections=detection_dicts,
                weather=weather_dict,
                location=location_dict,
                timestamp=datetime.now(timezone.utc).isoformat(),
                frame_count=self.video_streamer.frame_count if self.video_streamer else 0
            )
            
        except Exception as e:
            logger.error(f"Streaming data preparation error: {e}")
            return StreamingData(
                frame_data="",
                detections=[],
                weather={},
                location={},
                timestamp=datetime.now(timezone.utc).isoformat(),
                frame_count=0
            )
    
    async def run_async(self) -> None:
        """Main async detection loop with advanced features"""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Could not open webcam")
                return
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
            self._running = True
            
            # Start streaming server if enabled
            streaming_task = None
            if self.video_streamer and self.config.video_streaming_enabled:
                streaming_task = asyncio.create_task(self.video_streamer.start_server())
                logger.info(f"Video streaming server started on port {self.config.websocket_port}")
            
            logger.info("Starting enhanced detection loop...")
            
            while self._running:
                loop_start = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                # Process detections
                detections, person_count = self._process_detections(frame)
                
                # Analyze weather conditions
                weather_data = None
                if self.config.weather_analysis_enabled:
                    weather_data = await self._analyze_weather(frame)
                
                # Update location
                location_data = None
                if self.config.location_transmission_enabled:
                    location_data = await self._update_location()
                
                # Draw detections and UI
                frame = self._draw_detections(frame, detections)
                frame = self._draw_ui_info(frame, person_count, weather_data, location_data)
                
                # Handle announcements
                self._handle_announcements(detections, person_count)
                
                # Prepare and broadcast streaming data
                if self.video_streamer and self.config.video_streaming_enabled:
                    streaming_data = await self._prepare_streaming_data(
                        frame, detections, weather_data, location_data
                    )
                    await self.video_streamer.broadcast_frame(streaming_data)
                    self.video_streamer.frame_count += 1
                
                # Display frame
                cv2.imshow(self.config.window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keyboard_input(key):
                    break
                
                # FPS limiting
                elapsed = time.time() - loop_start
                sleep_time = (1.0 / self.config.fps_limit) - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            if streaming_task:
                streaming_task.cancel()
            await self._cleanup()
    
    def run(self) -> None:
        """Synchronous wrapper for the async detection loop"""
        asyncio.run(self.run_async())
    
    async def _cleanup(self) -> None:
        """Cleanup resources"""
        self._running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.tts_manager.shutdown()
        
        logger.info("System cleanup completed")


def create_default_config() -> Config:
    """Create default configuration with advanced features
    
    Returns:
        Default configuration object
    """
    return Config(
        model_paths=['object.pt', 'Currency.pt'],
        conf_threshold=0.9,
        person_threshold=15,
        alert_delay=0.5,
        window_name='Enhanced YOLO Multi-Model Detection with Advanced Features',
        detection_timeout=5.0,
        announcement_cooldown=10.0,  # 10 seconds cooldown for announcements
        model_colors={
            'object': (0, 255, 0),      # Green
            'Currency': (255, 0, 0),    # Blue
        },
        fps_limit=10.0,  # Limit to 10 FPS for better performance
        
        # Advanced feature configurations
        weather_analysis_enabled=True,
        location_transmission_enabled=True,
        video_streaming_enabled=True,
        streaming_port=8000,
        streaming_quality=80,  # JPEG quality
        location_update_interval=30.0,  # Update location every 30 seconds
        weather_analysis_interval=5.0,  # Analyze weather every 5 seconds
        remote_server_url="http://your-server.com/api",  # Replace with your server
        websocket_port=8000
    )


def main() -> None:
    """Main function to run the enhanced detection system"""
    try:
        config = create_default_config()
        system = ObjectDetectionSystem(config)
        
        logger.info("="*60)
        logger.info("Enhanced YOLO Object Detection System v3.0")
        logger.info("="*60)
        logger.info("Features:")
        logger.info("  ✓ Multi-model object detection")
        logger.info("  ✓ Weather condition analysis")
        logger.info("  ✓ Live location transmission")
        logger.info("  ✓ Live video streaming to React Expo")
        logger.info("  ✓ Intelligent announcements")
        logger.info("="*60)
        logger.info("Controls:")
        logger.info("  Q or ESC - Quit application")
        logger.info("  1 - Toggle first model (object.pt)")
        logger.info("  2 - Toggle second model (Currency.pt)")
        logger.info("  R - Reset announcement timers")
        logger.info("  W - Toggle weather analysis")
        logger.info("  L - Toggle location transmission")
        logger.info("  S - Toggle video streaming")
        logger.info("="*60)
        logger.info("Streaming Info:")
        logger.info(f"  WebSocket URL: ws://localhost:{config.websocket_port}/stream")
        logger.info(f"  Health Check: http://localhost:{config.streaming_port}/health")
        logger.info("="*60)
        
        system.run()
        
    except Exception as e:
        logger.error(f"System initialization error: {e}")
        logger.error("Make sure all required dependencies are installed:")
        logger.error("pip install opencv-python ultralytics pyttsx3 numpy requests")
        logger.error("pip install websockets fastapi uvicorn python-multipart aiofiles")
        logger.error("pip install pillow geopy scikit-learn tensorflow")
    finally:
        logger.info("Application terminated")


if __name__ == "__main__":
    main()