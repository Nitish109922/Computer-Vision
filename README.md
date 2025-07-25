# Enhanced Multi-Model YOLO Object Detection System with React Native WebSocket Support

## Overview

This project implements a real-time, multi-model object detection system using YOLO (You Only Look Once) models with advanced features tailored for mobile apps using React Native. The system streams annotated video frames over WebSocket, performs weather condition analysis from video input, tracks live location, and uses text-to-speech for alerts.

**Version**: 3.1
**Author**: Team 31

## Key Features

### ✅ Multi-Model YOLO Detection

* Supports loading and switching between multiple YOLOv8 models (e.g., object.pt, currency.pt).

### 🚀 WebSocket Streaming (React Native Compatible)

* Streams real-time detections, frames, weather data, and location info using WebSocket.

### 🌧 Weather Analysis from Video Feed

* Classifies real-world weather conditions (clear, rainy, foggy, etc.) using computer vision.

### 📍 Live Location Tracking

* Tracks and updates GPS or IP-based location, used in alert context and overlay.

### 💡 Intelligent TTS Alert System

* Announces new detections or crowd warnings via a thread-safe text-to-speech module.

### ⏱ Real-Time Performance Metrics

* Displays FPS, object counts, weather info, and WebSocket client stats directly on the video window.

### 🚨 Alert Throttling & Management

* Prevents repeated announcements with customizable cooldown periods.

### ⚖ Robust Error Handling & Logging

* All components wrapped with try/except and detailed logs to handle runtime issues.

---

## System Requirements

### Python Version

* Python 3.8+

### Required Python Packages

Install all dependencies using pip:

```bash
pip install opencv-python ultralytics pyttsx3 numpy requests websockets asyncio
pip install fastapi uvicorn python-multipart aiofiles pillow geopy
pip install opencv-contrib-python scikit-learn tensorflow
```

---

## Configuration

Modify `create_default_config()` to customize system behavior:

* **Models**: Add/Remove YOLO `.pt` files.
* **Thresholds**: Detection confidence and crowd alert.
* **Streaming**: WebSocket host, port, JPEG quality.
* **Intervals**: Location/weather update intervals.

Example snippet:

```python
Config(
    model_paths=["object.pt", "Currency.pt"],
    conf_threshold=0.5,
    person_threshold=2,
    websocket_host="0.0.0.0",
    websocket_port=8765,
    streaming_quality=80,
    location_update_interval=30.0,
    weather_analysis_interval=5.0
)
```

---

## Usage Instructions

### 1. Run the Script

```bash
python your_script_name.py
```

### 2. Controls During Execution

* Press `q` to quit
* Press `s` to take a screenshot
* Press `p` to pause/unpause detection

### 3. React Native Client Setup

Connect your React Native app using:

```js
const socket = new WebSocket("ws://<your_server_ip>:8765");
```

* Ensure device and server are on the same network.

---

## Output Data Structure

**WebSocket Message Format**:

```json
{
  "type": "frame",
  "image": "data:image/jpeg;base64,...",
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.98,
      "bbox": { "x1": 12, "y1": 34, "x2": 128, "y2": 256 },
      "model": "object",
      "color": {"r": 0, "g": 255, "b": 0}
    }
  ],
  "weather": {
    "condition": "clear",
    "confidence": 0.93,
    "brightness": 179.4,
    "contrast": 12.1
  },
  "location": {
    "latitude": 12.9716,
    "longitude": 77.5946,
    "address": "Bangalore, Karnataka, India",
    "method": "ip"
  }
}
```

---

## Future Improvements

* Add GPS-based fallback for mobile clients.
* Integrate cloud-based logging.
* Add YOLOv9 support when available.

---

## Troubleshooting

| Issue                        | Possible Fix                                   |
| ---------------------------- | ---------------------------------------------- |
| WebSocket not connecting     | Verify port 8765 is open, check firewall rules |
| No detection                 | Check model paths and confidence threshold     |
| Weather/Location not showing | Ensure internet access and proper permissions  |

---

## License

MIT License. Free for personal and commercial use.

---

## Contact

For technical queries, contact the author or raise a GitHub issue (if applicable).
