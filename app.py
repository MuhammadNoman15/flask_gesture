#!/usr/bin/env python3
"""
Flask Web Application for ASL Gesture Recognition
Provides a modern web interface for real-time gesture recognition
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import torch
import numpy as np
import json
import time
import threading
from datetime import datetime
import os
import sys
from typing import Optional, Dict, List
import base64

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.tracking.hand_tracker import HandTracker
from models.gesture_recognition.gesture_model import GestureRecognitionModel

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asl_gesture_recognition_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

class WebGestureRecognizer:
    def __init__(self, model_path: str):
        """Initialize the web-based gesture recognizer"""
        self.model_path = model_path
        self.hand_tracker = None  # Will be created when camera starts
        self.is_running = False
        self.cap = None
        self.camera_id = 0
        self._camera_thread = None
        self._stop_event = threading.Event()
        
        # Load the gesture recognition model
        print("📦 Loading model...")
        self.gesture_model = GestureRecognitionModel.load_model(
            model_path,
            input_size=21,
            hidden_size=128,
            num_classes=68,
            num_layers=2,
            dropout=0.2
        )
        
        # Load class names
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            if 'class_names' in checkpoint:
                class_names_dict = checkpoint['class_names']
            else:
                class_names_path = os.path.join(os.path.dirname(model_path), 
                                              'latest_visual_model_class_names.json')
                with open(class_names_path, 'r') as f:
                    class_names_dict = json.load(f)
            
            # Create mapping: index -> name (convert string keys to integers)
            self.index_to_name = {int(k): v for k, v in class_names_dict.items()}
            print(f"✅ Loaded {len(class_names_dict)} gesture classes")
            
        except Exception as e:
            print(f"❌ Error loading class names: {e}")
            self.index_to_name = {}
        
        # Recognition settings
        self.gesture_buffer = []
        self.buffer_size = 5
        self.confidence_threshold = 0.1
        self.current_gesture = None
        self.gesture_confidence = 0.0
        
        # Store recent predictions
        self.recent_predictions = []
        self.max_recent_predictions = 10
        
        # Don't pre-initialize HandTracker to avoid resource conflicts
        # HandTracker will be created when camera starts for cleaner initialization
        self.hand_tracker = None
        self._camera_optimized = False  # Flag for deferred camera optimization
        print("✅ System ready for camera initialization!")
        
    def start_camera(self, camera_id: int = 0):
        """Start the camera feed - optimized for fast startup"""
        # Stop any existing camera first
        if self.is_running:
            self.stop_camera()
            time.sleep(0.1)  # Reduced from 0.5s to 0.1s
        
        # Reset stop event
        self._stop_event.clear()
        self.camera_id = camera_id
        
        # Start camera initialization in background thread for instant response
        def _async_camera_init():
            try:
                print(f"🔧 Starting camera initialization for camera {camera_id}...")
                
                # Single-step camera initialization to minimize blinking
                if os.name == 'nt':  # Windows
                    self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(camera_id)
                
                print(f"📷 Camera opened: {self.cap.isOpened()}")
                if not self.cap.isOpened():
                    print("❌ Failed to open camera")
                    socketio.emit('camera_error', {'message': 'Failed to open camera'})
                    return
                
                # Minimal initialization - avoid ANY property changes during init to prevent blinking
                print("✅ Camera opened successfully!")
                print("⚡ Deferring camera optimization to first frame for zero-blink startup")
                
                # Create HandTracker after camera is fully configured
                print("🤖 Creating fresh HandTracker instance...")
                self.hand_tracker = HandTracker(feature_method=7)
                print("✅ HandTracker ready!")
                
                # Clear buffers
                self.gesture_buffer = []
                self.current_gesture = None
                self.gesture_confidence = 0.0
                
                # Flag to apply camera settings on first frame
                self._camera_optimized = False
                
                print("🚀 Setting is_running = True")
                self.is_running = True
                
                # Emit success after actual initialization
                print("📡 Emitting camera_ready event")
                socketio.emit('camera_ready', {'message': 'Camera ready'})
                
                # Start the main camera loop
                print("🎬 Starting camera loop...")
                self._camera_loop()
                
            except Exception as e:
                print(f"❌ Exception in camera initialization: {e}")
                import traceback
                traceback.print_exc()
                socketio.emit('camera_error', {'message': f'Camera initialization failed: {str(e)}'})
        
        # Start initialization in background
        self._camera_thread = threading.Thread(target=_async_camera_init, daemon=True)
        self._camera_thread.start()
        
        # Return immediately for fast UI response
        return {"success": True, "message": "Camera starting..."}
    
    def stop_camera(self):
        """Stop the camera feed"""
        self.is_running = False
        self._stop_event.set()
        
        # Wait for camera thread to finish
        if self._camera_thread and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=2.0)
        
        # Safely release camera
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                print(f"Warning: Error releasing camera: {e}")
            finally:
                self.cap = None
        
        # Clear resources
        self.hand_tracker = None
        self.gesture_buffer = []
        self.current_gesture = None
        self.gesture_confidence = 0.0
        self._camera_optimized = False  # Reset optimization flag for next start
        
        return {"success": True, "message": "Camera stopped"}
    
    def _camera_loop(self):
        """Main camera processing loop"""
        print("📹 Camera loop started")
        print(f"🔍 Debug: is_running={self.is_running}, cap={self.cap is not None}, hand_tracker={self.hand_tracker is not None}")
        frame_count = 0
        
        try:
            while self.is_running and not self._stop_event.is_set():
                if not self.cap or not self.hand_tracker:
                    print(f"❌ Camera loop exiting: cap={self.cap is not None}, hand_tracker={self.hand_tracker is not None}")
                    break
                    
                ret, frame = self.cap.read()
                if not ret:
                    print("Warning: Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Apply camera optimization on first successful frame to avoid init blinking
                if not getattr(self, '_camera_optimized', False):
                    print("🎯 Applying deferred camera optimization...")
                    
                    # Apply settings efficiently - only change if needed
                    try:
                        current_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        current_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        current_fps = self.cap.get(cv2.CAP_PROP_FPS)
                        
                        # Only set if different to minimize resets
                        if current_width != 640:
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        if current_height != 480:
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        if current_fps != 30:
                            self.cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        # Set buffer size for low latency
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        print(f"✅ Camera optimized: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(self.cap.get(cv2.CAP_PROP_FPS))}fps")
                        self._camera_optimized = True
                        
                    except Exception as e:
                        print(f"Warning: Camera optimization failed: {e}")
                        self._camera_optimized = True  # Don't keep trying
                
                try:
                    # Process frame for gesture recognition
                    processed_frame, prediction = self._process_frame(frame)
                    
                    # Convert frame to base64 for web transmission
                    _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Emit frame and prediction to web clients
                    socketio.emit('frame_update', {
                        'frame': frame_base64,
                        'prediction': prediction,
                        'timestamp': time.time(),
                        'frame_count': frame_count
                    })
                    
                except Exception as e:
                    print(f"Warning: Frame processing error: {e}")
                    # Continue processing other frames
                
                # Control frame rate
                if self._stop_event.wait(1/30):  # ~30 FPS with early exit
                    break
                    
        except Exception as e:
            print(f"Error in camera loop: {e}")
        finally:
            print("📹 Camera loop ended")
            self.is_running = False
    
    def _process_frame(self, frame: np.ndarray) -> tuple:
        """Process frame for gesture recognition"""
        processed_frame = frame.copy()
        prediction = None
        
        try:
            # Detect hands in frame
            processed_frame, hand_data = self.hand_tracker.detect_hands(frame)
            
            if hand_data:
                # Extract features
                features = self.hand_tracker.get_hand_pose_features(hand_data)
                
                if features is not None:
                    # Add to buffer
                    self.gesture_buffer.append(features)
                    
                    # Keep buffer at fixed size
                    if len(self.gesture_buffer) > self.buffer_size:
                        self.gesture_buffer.pop(0)
                    
                    # Make prediction when buffer is full
                    if len(self.gesture_buffer) == self.buffer_size:
                        try:
                            sequence = np.array(self.gesture_buffer)
                            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
                            
                            with torch.no_grad():
                                outputs, attention = self.gesture_model(sequence_tensor)
                                probabilities = torch.softmax(outputs, dim=1)
                                
                                # Get prediction
                                confidence, predicted_idx = torch.max(probabilities, 1)
                                self.gesture_confidence = confidence.item()
                                predicted_gesture = self.index_to_name.get(predicted_idx.item(), "UNKNOWN")
                                
                                # Update current gesture if confidence is good
                                if self.gesture_confidence > self.confidence_threshold:
                                    self.current_gesture = predicted_gesture
                                    
                                    # Add to recent predictions
                                    prediction_data = {
                                        'gesture': predicted_gesture,
                                        'confidence': round(self.gesture_confidence * 100, 1),
                                        'timestamp': datetime.now().strftime("%I:%M:%S %p")
                                    }
                                    
                                    # Only add if different from last prediction or significantly higher confidence
                                    if (not self.recent_predictions or 
                                        self.recent_predictions[0]['gesture'] != predicted_gesture or
                                        self.gesture_confidence > 0.3):
                                        
                                        self.recent_predictions.insert(0, prediction_data)
                                        if len(self.recent_predictions) > self.max_recent_predictions:
                                            self.recent_predictions.pop()
                                    
                                    prediction = prediction_data
                                
                        except Exception as e:
                            print(f"❌ Model prediction error: {e}")
            else:
                # Clear buffer if no hands detected
                self.gesture_buffer.clear()
                self.current_gesture = None
                
        except Exception as e:
            print(f"❌ Hand tracking error: {e}")
            # Use original frame if hand tracking fails
            processed_frame = frame.copy()
        
        # Add visualizations to frame
        self._add_visualizations(processed_frame)
        
        return processed_frame, prediction
    
    def _add_visualizations(self, frame: np.ndarray):
        """Add visualizations to frame"""
        # Text overlays removed for cleaner video feed
        # The gesture recognition results are displayed in the web UI instead
        pass

# Initialize the gesture recognizer
model_path = os.path.join(os.path.dirname(__file__), 'models', 'gesture_recognition', 'latest_visual_model.pth')
recognizer = WebGestureRecognizer(model_path)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start camera endpoint"""
    camera_id = request.json.get('camera_id', 0)
    result = recognizer.start_camera(camera_id)
    return jsonify(result)

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera endpoint"""
    result = recognizer.stop_camera()
    return jsonify(result)

@app.route('/get_predictions')
def get_predictions():
    """Get recent predictions"""
    return jsonify({
        'recent_predictions': recognizer.recent_predictions,
        'current_gesture': recognizer.current_gesture,
        'current_confidence': round(recognizer.gesture_confidence * 100, 1) if recognizer.gesture_confidence else 0,
        'total_gestures': len(recognizer.index_to_name)
    })

@app.route('/clear_predictions', methods=['POST'])
def clear_predictions():
    """Clear recent predictions"""
    recognizer.recent_predictions.clear()
    return jsonify({"success": True, "message": "Predictions cleared"})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status', {'message': 'Connected to ASL Gesture Recognition'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    print("🚀 Starting ASL Gesture Recognition Web App...")
    print("📱 Open your browser to: http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)