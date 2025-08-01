import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional, Dict

class HandTracker:
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 feature_method: int = 7):  # Only one method now
        """
        Simple hand tracker using MediaPipe for accurate gesture recognition.
        
        Args:
            static_image_mode: If True, treats input as a single image
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
            feature_method: Feature extraction method (kept for compatibility)
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1  # Use standard model for better accuracy
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def detect_hands(self, frame: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect hands in the given frame.
        
        Args:
            frame: Input image/frame (BGR format)
            draw: Whether to draw hand landmarks on the frame
            
        Returns:
            Tuple containing:
            - Processed frame with hand landmarks drawn (if draw=True)
            - List of detected hand data dictionaries
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Prepare output frame
        output_frame = frame.copy()
        hand_data = []
        
        # Process detected hands
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks if requested
                if draw:
                    self.mp_draw.draw_landmarks(
                        output_frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
                
                # Extract landmark coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                
                # Determine handedness
                handedness = "Right"  # Default
                if results.multi_handedness and idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[idx].classification[0].label
                
                # Store hand data
                hand_info = {
                    'landmarks': np.array(landmarks),
                    'handedness': handedness,
                    'raw_landmarks': hand_landmarks
                }
                hand_data.append(hand_info)
        
        return output_frame, hand_data
    
    def get_hand_pose_features(self, hand_data: List[Dict]) -> Optional[np.ndarray]:
        """
        Extract simple, reliable features from hand pose data.
        Improved to better match training data format.
        
        Args:
            hand_data: List of hand data dictionaries from detect_hands()
            
        Returns:
            Feature vector as numpy array, or None if no hands detected
        """
        if not hand_data:
            return None
        
        # Use the first detected hand (can be extended for two-hand gestures)
        hand = hand_data[0]
        landmarks = hand['landmarks']
        
        # Improved feature extraction - use normalized relative positions
        # Get wrist as reference point (landmark 0)
        wrist = landmarks[0]
        features = []
        
        # Get all 21 landmarks (0-20)
        for i in range(21):
            if i < len(landmarks):
                point = landmarks[i]
                # Calculate relative position from wrist (x and y only, ignore z)
                relative_x = point[0] - wrist[0]
                relative_y = point[1] - wrist[1]
                
                # Use Euclidean distance from wrist as the feature
                # This creates a scale-invariant representation
                distance = np.sqrt(relative_x**2 + relative_y**2)
                features.append(distance)
            else:
                features.append(0.0)
        
        # Normalize features to [0, 1] range for better model performance
        features = np.array(features, dtype=np.float32)
        if np.max(features) > 0:
            features = features / np.max(features)
        
        # Ensure we have exactly 21 features
        return features[:21]
    
    def get_hand_pose_features_optimized(self, hand_data: List[Dict]) -> Optional[np.ndarray]:
        """
        Optimized feature extraction (same as standard for simplicity).
        """
        return self.get_hand_pose_features(hand_data)
    
    def get_hand_confidence(self, hand_data: List[Dict]) -> float:
        """Get the confidence of hand detection."""
        if not hand_data:
            return 0.0
        # Since MediaPipe doesn't provide confidence directly, return 1.0 for detected hands
        return 1.0
    
    def get_dominant_hand(self, hand_data: List[Dict]) -> Optional[str]:
        """Get the handedness of the most confident hand."""
        if not hand_data:
            return None
        
        # Return the first hand's handedness (most reliable)
        return hand_data[0]['handedness']
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'hands'):
            self.hands.close() 