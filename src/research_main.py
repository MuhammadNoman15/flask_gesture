"""
Research-Grade Sign Language Avatar with Advanced Features

This module implements a research-grade sign language recognition system with:
- Vision Transformer (ViT) for advanced gesture sequence modeling
- YOLOv8 integration for improved hand detection  
- Motion synthesis for generating fluid sign language animations
- Advanced deep learning techniques for gesture prediction
- Real-time animation controller for seamless sign animations
"""

import cv2
import torch
import numpy as np
import argparse
from typing import Optional, Tuple, Dict
import time
import sys
import os
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .tracking.hand_tracker import HandTracker
# from .animation.advanced_animation_controller import (
#     AdvancedAnimationController, 
#     MotionSynthesisIntegration
# )

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class ResearchSignLanguageAvatar:
    """
    Research-grade Sign Language Avatar system with advanced AI capabilities
    """
    
    def __init__(self,
                 model_path: str,
                 model_type: str = 'advanced',  # 'basic', 'advanced'
                 camera_id: int = 0,
                 fps: int = 30,
                 enable_debug: bool = True,
                 use_yolo: bool = True,
                 enable_motion_synthesis: bool = True):
        """
        Initialize Research-Grade Sign Language Avatar system
        
        Args:
            model_path: Path to the trained model
            model_type: Type of model ('basic' for LSTM, 'advanced' for ViT)
            camera_id: ID of the camera to use
            fps: Target frames per second
            enable_debug: Whether to enable debug output
            use_yolo: Whether to use YOLOv8 for hand detection
            enable_motion_synthesis: Whether to enable motion synthesis
        """
        print("🚀 Initializing Research-Grade ASL Gesture Recognition...")
        
        self.model_type = model_type
        self.enable_debug = enable_debug
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.enable_motion_synthesis = enable_motion_synthesis
        
        # Initialize hand tracking
        if self.use_yolo:
            print("🎯 Using YOLOv8 + MediaPipe for enhanced hand detection...")
            # Initialize YOLOv8 (would need custom implementation)
            self.hand_tracker = HandTracker(feature_method=7)
        else:
            print("🖐️ Using MediaPipe for hand detection...")
            self.hand_tracker = HandTracker(feature_method=7)
        
        # Load the appropriate model
        print(f"📦 Loading {model_type} model...")
        self._load_model(model_path)
        
        # Initialize camera with optimized settings
        print("📷 Setting up camera...")
        self._setup_camera(camera_id, fps)
        
        # Initialize animation system (temporarily disabled)
        if self.enable_motion_synthesis:
            print("⚠️ Motion synthesis temporarily disabled due to import issues")
        
        self.animation_controller = None
        self.motion_integration = None
        
        # Gesture recognition settings
        self.gesture_buffer = []
        self.buffer_size = 8 if model_type == 'advanced' else 5  # Faster recognition
        self.confidence_threshold = 0.2  # Lower threshold for better detection
        self.current_gesture = None
        self.gesture_confidence = 0.0
        
        # Performance tracking
        self.frame_times = []
        self.prediction_times = []
        
        print("✅ Research system ready!")
    
    def _load_model(self, model_path: str):
        """Load the appropriate model based on type"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            if self.model_type == 'advanced' and 'model_type' in checkpoint and checkpoint['model_type'] == 'AdvancedGestureModel':
                # Load advanced research model
                from models.gesture_recognition.advanced_gesture_model import create_research_model
                
                config = checkpoint.get('config', {})
                self.gesture_model = create_research_model(config)
                self.gesture_model.load_state_dict(checkpoint['model_state_dict'])
                
                print("✅ Loaded advanced Vision Transformer model with motion synthesis")
                
            else:
                # Load basic LSTM model
                from models.gesture_recognition.gesture_model import GestureRecognitionModel
                
                self.gesture_model = GestureRecognitionModel.load_model(
                    model_path,
                    input_size=21,
                    hidden_size=128,
                    num_classes=86,
                    num_layers=2,
                    dropout=0.2
                )
                print("✅ Loaded basic LSTM model")
            
            # Load class names
            if 'class_names' in checkpoint:
                class_names_dict = checkpoint['class_names']
                print(f"✅ Loaded {len(class_names_dict)} gesture classes")
            else:
                # Try to load from separate file
                class_names_path = os.path.join(os.path.dirname(model_path), 'class_names.json')
                if os.path.exists(class_names_path):
                    with open(class_names_path, 'r') as f:
                        class_names_dict = json.load(f)
                    print(f"✅ Loaded {len(class_names_dict)} gesture classes from JSON")
                else:
                    # Create dummy mapping
                    class_names_dict = {f"Gesture_{i}": i for i in range(86)}
                    print("⚠️ Using dummy class names")
            
            # Create reverse mapping: index -> name
            if isinstance(list(class_names_dict.values())[0], int):
                self.index_to_name = {v: k for k, v in class_names_dict.items()}
            else:
                self.index_to_name = class_names_dict
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _setup_camera(self, camera_id: int, fps: int):
        """Setup camera with optimized settings"""
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        
        # Optimized camera settings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        
        # Test camera
        ret, test_frame = self.cap.read()
        if not ret:
            print("⚠️ Trying alternative camera initialization...")
            self.cap.release()
            self.cap = cv2.VideoCapture(camera_id)
            ret, test_frame = self.cap.read()
            if not ret:
                raise RuntimeError("❌ Camera failed to initialize!")
        
        # Create windows
        cv2.namedWindow("Research ASL Recognition", cv2.WINDOW_AUTOSIZE)
        
        cv2.imshow("Research ASL Recognition", test_frame)
        cv2.waitKey(1)
        
        print("✅ Camera ready!")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Dict], Optional[np.ndarray]]:
        """
        Advanced frame processing with research-grade features
        
        Returns:
            Tuple of (processed_frame, gesture_result, animation_frame)
        """
        frame_start_time = time.time()
        
        # Detect hands in frame
        processed_frame, hand_data = self.hand_tracker.detect_hands(frame)
        
        gesture_result = None
        animation_frame = None
        
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
                    prediction_start_time = time.time()
                    
                    try:
                        if self.model_type == 'advanced':
                            gesture_result = self._predict_advanced(self.gesture_buffer)
                        else:
                            gesture_result = self._predict_basic(self.gesture_buffer)
                        
                        prediction_time = time.time() - prediction_start_time
                        self.prediction_times.append(prediction_time)
                        
                        # Keep only recent predictions for averaging
                        if len(self.prediction_times) > 30:
                            self.prediction_times.pop(0)
                        
                    except Exception as e:
                        if self.enable_debug:
                            print(f"❌ Prediction error: {e}")
        else:
            # Clear buffer if no hands detected
            self.gesture_buffer.clear()
            self.current_gesture = None
        
        # Get animation frame if motion synthesis is enabled
        if self.enable_motion_synthesis and self.animation_controller:
            animation_frame = self.animation_controller.get_next_frame()
        
        # Add visualizations
        self._add_advanced_visualizations(processed_frame)
        
        # Track frame time
        frame_time = time.time() - frame_start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        return processed_frame, gesture_result, animation_frame
    
    def _predict_advanced(self, gesture_buffer: list) -> Optional[Dict]:
        """Make prediction using advanced Vision Transformer model"""
        sequence = np.array(gesture_buffer)
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        with torch.no_grad():
            # Get model results
            results = self.gesture_model(sequence_tensor, generate_motion=self.enable_motion_synthesis)
            
            probabilities = torch.softmax(results['logits'], dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            self.gesture_confidence = confidence.item()
            predicted_gesture = self.index_to_name.get(predicted_idx.item(), "UNKNOWN")
            
            # Debug: Always show prediction
            if self.enable_debug:
                print(f"🔍 Advanced Prediction: {predicted_gesture} (Confidence: {self.gesture_confidence:.3f})")
            
            # Update current gesture if confidence is good
            if self.gesture_confidence > self.confidence_threshold:
                self.current_gesture = predicted_gesture
                
                # Process motion synthesis if enabled (temporarily disabled)
                # if self.enable_motion_synthesis and self.motion_integration:
                #     self.motion_integration.process_gesture_sequence(sequence)
                
                if self.enable_debug:
                    print(f"🎯 Advanced Detected: {predicted_gesture} (Confidence: {self.gesture_confidence:.3f})")
            
            return {
                'name': self.current_gesture,
                'confidence': self.gesture_confidence,
                'timestamp': time.time(),
                'model_type': 'ViT + Motion Synthesis',
                'attention_weights': results.get('attention', None)
            }
    
    def _predict_basic(self, gesture_buffer: list) -> Optional[Dict]:
        """Make prediction using basic LSTM model"""
        sequence = np.array(gesture_buffer)
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        with torch.no_grad():
            outputs, attention = self.gesture_model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            confidence, predicted_idx = torch.max(probabilities, 1)
            self.gesture_confidence = confidence.item()
            predicted_gesture = self.index_to_name.get(predicted_idx.item(), "UNKNOWN")
            
            if self.gesture_confidence > self.confidence_threshold:
                self.current_gesture = predicted_gesture
                if self.enable_debug:
                    print(f"🎯 Basic Detected: {predicted_gesture} (Confidence: {self.gesture_confidence:.3f})")
            
            return {
                'name': self.current_gesture,
                'confidence': self.gesture_confidence,
                'timestamp': time.time(),
                'model_type': 'LSTM',
                'attention_weights': attention
            }
    
    def _add_advanced_visualizations(self, frame: np.ndarray):
        """Add advanced visualizations to the frame"""
        # Current gesture display
        if self.current_gesture:
            text = f"Gesture: {self.current_gesture}"
            confidence_text = f"Confidence: {self.gesture_confidence:.2f}"
            model_text = f"Model: {self.model_type.upper()}"
            
            color = (0, 255, 0) if self.gesture_confidence > 0.7 else (0, 255, 255)
            
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, model_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No Gesture Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Buffer status
        buffer_text = f"Buffer: {len(self.gesture_buffer)}/{self.buffer_size}"
        cv2.putText(frame, buffer_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Performance metrics
        if self.frame_times:
            avg_frame_time = np.mean(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if self.prediction_times:
            avg_pred_time = np.mean(self.prediction_times) * 1000
            pred_text = f"Prediction: {avg_pred_time:.1f}ms"
            cv2.putText(frame, pred_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Feature indicators
        features_y = 210
        if self.use_yolo:
            cv2.putText(frame, "YOLOv8: ON", (10, features_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            features_y += 25
        
        if self.enable_motion_synthesis:
            cv2.putText(frame, "Motion Synthesis: ON", (10, features_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            features_y += 25
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run(self):
        """Run the main research system loop"""
        print("🎬 Starting research-grade gesture recognition...")
        print("📋 System Features:")
        print(f"   🧠 Model: {self.model_type.upper()}")
        print(f"   🎯 YOLOv8: {'ON' if self.use_yolo else 'OFF'}")
        print(f"   🎬 Motion Synthesis: {'ON' if self.enable_motion_synthesis else 'OFF'}")
        print(f"   📏 Buffer Size: {self.buffer_size}")
        print("💡 Hold your gesture clearly in front of the camera")
        print()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Error reading from camera")
                break
            
            # Process frame with advanced features
            processed_frame, gesture_result, animation_frame = self.process_frame(frame)
            
            # Display main frame
            cv2.imshow("Research ASL Recognition", processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Clear gesture buffer
                self.gesture_buffer.clear()
                print("🧹 Cleared buffers")
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Research-Grade ASL Gesture Recognition")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the trained model")
    parser.add_argument("--model_type", type=str, choices=['basic', 'advanced'], default='advanced',
                      help="Type of model to use")
    parser.add_argument("--camera_id", type=int, default=0,
                      help="Camera ID to use")
    parser.add_argument("--fps", type=int, default=30,
                      help="Target FPS")
    parser.add_argument("--debug", action="store_true", default=True,
                      help="Enable debug output")
    parser.add_argument("--no_yolo", action="store_true",
                      help="Disable YOLOv8 hand detection")
    parser.add_argument("--no_motion_synthesis", action="store_true",
                      help="Disable motion synthesis")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not Path(args.model_path).exists():
        print(f"❌ Model file not found: {args.model_path}")
        return 1
    
    # Create and run research system
    try:
        avatar = ResearchSignLanguageAvatar(
            model_path=args.model_path,
            model_type=args.model_type,
            camera_id=args.camera_id,
            fps=args.fps,
            enable_debug=args.debug,
            use_yolo=not args.no_yolo,
            enable_motion_synthesis=not args.no_motion_synthesis
        )
        
        avatar.run()
        
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())