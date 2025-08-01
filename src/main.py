import cv2
import torch
import numpy as np
import argparse
from typing import Optional, Tuple, Dict
import time
import sys
import os
import json

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .tracking.hand_tracker import HandTracker
from .animation.animation_synthesizer import AnimationSynthesizer
from models.gesture_recognition.gesture_model import GestureRecognitionModel

class SignLanguageAvatar:
    def __init__(self,
                 model_path: str,
                 camera_id: int = 0,
                 fps: int = 30,
                 enable_debug: bool = True):
        """
        Simple Sign Language Avatar system for accurate gesture recognition.
        
        Args:
            model_path: Path to the trained gesture recognition model
            camera_id: ID of the camera to use
            fps: Target frames per second
            enable_debug: Whether to enable debug output
        """
        print("🚀 Initializing ASL Gesture Recognition...")
        
        # Initialize components with simple settings
        self.hand_tracker = HandTracker(feature_method=7)  # Use standard method
        
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
                print(f"✅ Loaded {len(class_names_dict)} gesture classes")
            else:
                class_names_path = os.path.join(os.path.dirname(model_path), 'class_names.json')
                with open(class_names_path, 'r') as f:
                    class_names_dict = json.load(f)
                print(f"✅ Loaded {len(class_names_dict)} gesture classes from JSON")
        except Exception as e:
            print(f"❌ Error loading class names: {e}")
            return
            
        # Create mapping: index -> name (convert string keys to integers)
        self.index_to_name = {int(k): v for k, v in class_names_dict.items()}
        
        # Debug: Show first few mappings
        if enable_debug:
            print("🔍 Class mappings (first 5):")
            for i in sorted(list(self.index_to_name.keys())[:5]):
                print(f"  {i}: {self.index_to_name[i]}")
        
        # Initialize camera with optimized settings for fast startup
        print("📷 Setting up camera...")
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)  # Use DirectShow on Windows for faster init
        
        # Fast camera settings - balanced for speed and quality
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Re-enable autofocus for clear image (keep auto-exposure disabled for speed)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto exposure for speed
        # Note: Removed AUTOFOCUS=0 to allow camera to focus properly
        
        # Additional quality settings
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
        
        # Test camera immediately and create window
        print("🎬 Testing camera and creating window...")
        ret, test_frame = self.cap.read()
        if not ret:
            print("⚠️ Trying alternative camera initialization...")
            self.cap.release()
            self.cap = cv2.VideoCapture(camera_id)  # Fallback to default
            ret, test_frame = self.cap.read()
            if not ret:
                raise RuntimeError("❌ Camera failed to initialize!")
        
        # Create OpenCV window immediately after camera test
        cv2.namedWindow("ASL Gesture Recognition", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ASL Gesture Recognition", test_frame)
        cv2.waitKey(1)  # Process window events immediately
        
        print("✅ Camera and window ready!")
        
        # Simple gesture recognition settings
        self.enable_debug = enable_debug
        self.gesture_buffer = []
        self.buffer_size = 5  # Reduced from 8 for faster detection (5 frames = ~166ms at 30fps)
        self.confidence_threshold = 0.1  # Lowered from 0.4 to catch more predictions
        self.current_gesture = None
        self.gesture_confidence = 0.0
        
        # Animation synthesizer
        self.animation_synthesizer = AnimationSynthesizer(fps=fps, smoothing_window=5)
        
        print("✅ System ready!")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Simple frame processing for accurate gesture recognition.
        """
        # Detect hands in frame
        processed_frame, hand_data = self.hand_tracker.detect_hands(frame)
        
        gesture_result = None
        
        if hand_data:
            # Extract features using standard method
            features = self.hand_tracker.get_hand_pose_features(hand_data)
            
            if features is not None:
                # Add to buffer
                self.gesture_buffer.append(features)
                
                # Keep buffer at fixed size
                if len(self.gesture_buffer) > self.buffer_size:
                    self.gesture_buffer.pop(0)
                
                # Make prediction when buffer is full
                if len(self.gesture_buffer) == self.buffer_size:
                    # Create sequence for model
                    sequence = np.array(self.gesture_buffer)
                    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
                    
                    try:
                        with torch.no_grad():
                            outputs, attention = self.gesture_model(sequence_tensor)
                            probabilities = torch.softmax(outputs, dim=1)
                            
                            # Get prediction
                            confidence, predicted_idx = torch.max(probabilities, 1)
                            self.gesture_confidence = confidence.item()
                            predicted_gesture = self.index_to_name.get(predicted_idx.item(), "UNKNOWN")
                            
                            # Debug: Show ALL predictions
                            if self.enable_debug:
                                print(f"🔍 Prediction: Index {predicted_idx.item()} -> {predicted_gesture} (Confidence: {self.gesture_confidence:.3f})")
                            
                            # Update current gesture if confidence is good
                            if self.gesture_confidence > self.confidence_threshold:
                                self.current_gesture = predicted_gesture
                                if self.enable_debug:
                                    print(f"🎯 Accepted: {predicted_gesture} (Confidence: {self.gesture_confidence:.3f})")
                            
                            gesture_result = {
                                'name': self.current_gesture,
                                'confidence': self.gesture_confidence,
                                'timestamp': time.time()
                            }
                            
                    except Exception as e:
                        if self.enable_debug:
                            print(f"❌ Prediction error: {e}")
        else:
            # Clear buffer if no hands detected
            self.gesture_buffer.clear()
            self.current_gesture = None
        
        # Add visualizations
        self._add_visualizations(processed_frame)
        
        # Get animation frame
        animation_frame = self.animation_synthesizer.get_next_frame()
        
        return processed_frame, animation_frame

    def _add_visualizations(self, frame: np.ndarray):
        """Add simple visualizations to the frame."""
        if self.current_gesture:
            # Display current gesture
            text = f"Gesture: {self.current_gesture}"
            confidence_text = f"Confidence: {self.gesture_confidence:.2f}"
            
            color = (0, 255, 0) if self.gesture_confidence > 0.7 else (0, 255, 255)
            
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(frame, "No Gesture Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Buffer status
        buffer_text = f"Buffer: {len(self.gesture_buffer)}/{self.buffer_size}"
        cv2.putText(frame, buffer_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self):
        """Run the main loop."""
        print("🎬 Starting gesture recognition...")
        print("📋 System can detect all 86 trained ASL gestures")
        print("💡 Hold your gesture clearly in front of the camera")
        print()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Error reading from camera")
                break
            
            # Process frame
            processed_frame, animation_frame = self.process_frame(frame)
            
            # Display frame
            cv2.imshow("ASL Gesture Recognition", processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Simple ASL Gesture Recognition")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the trained model")
    parser.add_argument("--camera_id", type=int, default=0,
                      help="Camera ID to use")
    parser.add_argument("--fps", type=int, default=30,
                      help="Target FPS")
    parser.add_argument("--debug", action="store_true", default=True,
                      help="Enable debug output")
    
    args = parser.parse_args()
    
    # Create and run system
    avatar = SignLanguageAvatar(
        model_path=args.model_path,
        camera_id=args.camera_id,
        fps=args.fps,
        enable_debug=args.debug
    )
    
    try:
        avatar.run()
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    finally:
        avatar.cleanup()

if __name__ == "__main__":
    main() 