"""
Advanced Animation Controller for Sign Language Motion Synthesis

This module implements a Python-based animation controller that creates
seamless, fluid sign language animations from motion synthesis predictions.
It integrates with the research model to produce realistic gesture animations.
"""

import numpy as np
import cv2
import torch
import json
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
import time
from collections import deque

class MotionSmoother:
    """
    Advanced motion smoothing for natural gesture animations
    """
    
    def __init__(self, 
                 smoothing_window: int = 10,
                 interpolation_factor: int = 3,
                 gaussian_sigma: float = 1.0):
        """
        Initialize motion smoother
        
        Args:
            smoothing_window: Size of smoothing window
            interpolation_factor: Factor for temporal interpolation
            gaussian_sigma: Sigma for Gaussian smoothing
        """
        self.smoothing_window = smoothing_window
        self.interpolation_factor = interpolation_factor
        self.gaussian_sigma = gaussian_sigma
        self.motion_buffer = deque(maxlen=smoothing_window)
    
    def smooth_motion_sequence(self, motion_sequence: np.ndarray) -> np.ndarray:
        """
        Smooth a motion sequence using advanced filtering techniques
        
        Args:
            motion_sequence: Input motion sequence (frames, landmarks, coordinates)
            
        Returns:
            Smoothed motion sequence
        """
        if len(motion_sequence) < 2:
            return motion_sequence
        
        # Apply Gaussian filtering along time axis
        smoothed = gaussian_filter1d(motion_sequence, sigma=self.gaussian_sigma, axis=0)
        
        # Temporal interpolation for smoother motion
        if self.interpolation_factor > 1:
            smoothed = self._interpolate_temporal(smoothed)
        
        return smoothed
    
    def _interpolate_temporal(self, motion_sequence: np.ndarray) -> np.ndarray:
        """
        Perform temporal interpolation to increase frame rate
        
        Args:
            motion_sequence: Input motion sequence
            
        Returns:
            Interpolated motion sequence
        """
        original_length = motion_sequence.shape[0]
        new_length = original_length * self.interpolation_factor
        
        # Create interpolation functions for each landmark coordinate
        interpolated = np.zeros((new_length, motion_sequence.shape[1], motion_sequence.shape[2]))
        
        old_indices = np.linspace(0, original_length - 1, original_length)
        new_indices = np.linspace(0, original_length - 1, new_length)
        
        for landmark_idx in range(motion_sequence.shape[1]):
            for coord_idx in range(motion_sequence.shape[2]):
                # Use cubic interpolation for smooth motion
                f = interpolate.interp1d(old_indices, motion_sequence[:, landmark_idx, coord_idx], 
                                       kind='cubic', bounds_error=False, fill_value='extrapolate')
                interpolated[:, landmark_idx, coord_idx] = f(new_indices)
        
        return interpolated
    
    def add_frame(self, frame_motion: np.ndarray) -> np.ndarray:
        """
        Add a frame to the buffer and return smoothed motion
        
        Args:
            frame_motion: Motion data for current frame
            
        Returns:
            Smoothed motion for current frame
        """
        self.motion_buffer.append(frame_motion)
        
        if len(self.motion_buffer) < self.smoothing_window:
            return frame_motion
        
        # Apply smoothing to buffer
        buffer_array = np.array(list(self.motion_buffer))
        smoothed_buffer = self.smooth_motion_sequence(buffer_array)
        
        # Return the middle frame (most stable)
        middle_idx = len(smoothed_buffer) // 2
        return smoothed_buffer[middle_idx]

class GestureTransition:
    """
    Handles smooth transitions between different gestures
    """
    
    def __init__(self, transition_frames: int = 10):
        """
        Initialize gesture transition handler
        
        Args:
            transition_frames: Number of frames for smooth transition
        """
        self.transition_frames = transition_frames
        self.current_gesture_motion = None
        self.target_gesture_motion = None
        self.transition_progress = 0.0
        self.in_transition = False
    
    def start_transition(self, 
                        current_motion: np.ndarray, 
                        target_motion: np.ndarray) -> None:
        """
        Start a smooth transition between gestures
        
        Args:
            current_motion: Current gesture motion
            target_motion: Target gesture motion
        """
        self.current_gesture_motion = current_motion
        self.target_gesture_motion = target_motion
        self.transition_progress = 0.0
        self.in_transition = True
    
    def get_transition_frame(self) -> Optional[np.ndarray]:
        """
        Get the next frame in the transition
        
        Returns:
            Interpolated motion frame or None if transition complete
        """
        if not self.in_transition:
            return None
        
        # Smooth interpolation using cosine easing
        alpha = 0.5 * (1 - np.cos(np.pi * self.transition_progress))
        
        # Interpolate between current and target motion
        interpolated_motion = (1 - alpha) * self.current_gesture_motion + alpha * self.target_gesture_motion
        
        # Update progress
        self.transition_progress += 1.0 / self.transition_frames
        
        if self.transition_progress >= 1.0:
            self.in_transition = False
        
        return interpolated_motion
    
    def is_transitioning(self) -> bool:
        """Check if currently in transition"""
        return self.in_transition

class AnimationRenderer:
    """
    Renders sign language animations with advanced visualization
    """
    
    def __init__(self, 
                 canvas_size: Tuple[int, int] = (800, 600),
                 background_color: Tuple[int, int, int] = (50, 50, 50)):
        """
        Initialize animation renderer
        
        Args:
            canvas_size: Size of animation canvas (width, height)
            background_color: Background color (B, G, R)
        """
        self.canvas_size = canvas_size
        self.background_color = background_color
        
        # Hand connections for drawing (MediaPipe format)
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        # Colors for different parts
        self.colors = {
            'landmarks': (0, 255, 255),  # Yellow
            'connections': (255, 255, 255),  # White
            'thumb': (255, 0, 0),  # Red
            'index': (0, 255, 0),  # Green
            'middle': (0, 0, 255),  # Blue
            'ring': (255, 0, 255),  # Magenta
            'pinky': (255, 255, 0)  # Cyan
        )
    
    def render_frame(self, 
                    motion_data: np.ndarray, 
                    gesture_info: Optional[Dict] = None) -> np.ndarray:
        """
        Render a single animation frame
        
        Args:
            motion_data: Motion data for hand landmarks (21, 3)
            gesture_info: Optional gesture information for display
            
        Returns:
            Rendered frame as BGR image
        """
        # Create canvas
        canvas = np.full((self.canvas_size[1], self.canvas_size[0], 3), 
                        self.background_color, dtype=np.uint8)
        
        if motion_data is None or motion_data.shape[0] < 21:
            return canvas
        
        # Convert normalized coordinates to canvas coordinates
        landmarks_2d = self._normalize_to_canvas(motion_data)
        
        # Draw hand connections
        self._draw_hand_connections(canvas, landmarks_2d)
        
        # Draw landmarks
        self._draw_landmarks(canvas, landmarks_2d)
        
        # Add gesture information if provided
        if gesture_info:
            self._draw_gesture_info(canvas, gesture_info)
        
        return canvas
    
    def _normalize_to_canvas(self, motion_data: np.ndarray) -> np.ndarray:
        """
        Convert normalized motion data to canvas coordinates
        
        Args:
            motion_data: Normalized motion data (21, 3)
            
        Returns:
            Canvas coordinates (21, 2)
        """
        # Use only x, y coordinates (ignore z)
        landmarks_2d = motion_data[:, :2]
        
        # Scale to canvas size with padding
        padding = 50
        canvas_width = self.canvas_size[0] - 2 * padding
        canvas_height = self.canvas_size[1] - 2 * padding
        
        # Normalize to [0, 1] range
        landmarks_2d = (landmarks_2d + 1) / 2  # Assuming input is in [-1, 1]
        
        # Scale to canvas coordinates
        landmarks_2d[:, 0] = landmarks_2d[:, 0] * canvas_width + padding
        landmarks_2d[:, 1] = landmarks_2d[:, 1] * canvas_height + padding
        
        return landmarks_2d.astype(int)
    
    def _draw_hand_connections(self, canvas: np.ndarray, landmarks: np.ndarray):
        """Draw hand connections"""
        for connection in self.hand_connections:
            start_point = tuple(landmarks[connection[0]])
            end_point = tuple(landmarks[connection[1]])
            cv2.line(canvas, start_point, end_point, self.colors['connections'], 2)
    
    def _draw_landmarks(self, canvas: np.ndarray, landmarks: np.ndarray):
        """Draw hand landmarks"""
        for i, point in enumerate(landmarks):
            # Different colors for different fingers
            if i == 0:  # Wrist
                color = (255, 255, 255)
                radius = 8
            elif i <= 4:  # Thumb
                color = self.colors['thumb']
                radius = 5
            elif i <= 8:  # Index
                color = self.colors['index']
                radius = 5
            elif i <= 12:  # Middle
                color = self.colors['middle']
                radius = 5
            elif i <= 16:  # Ring
                color = self.colors['ring']
                radius = 5
            else:  # Pinky
                color = self.colors['pinky']
                radius = 5
            
            cv2.circle(canvas, tuple(point), radius, color, -1)
            cv2.circle(canvas, tuple(point), radius + 1, (0, 0, 0), 1)
    
    def _draw_gesture_info(self, canvas: np.ndarray, gesture_info: Dict):
        """Draw gesture information on canvas"""
        if 'name' in gesture_info:
            text = f"Gesture: {gesture_info['name']}"
            cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if 'confidence' in gesture_info:
            text = f"Confidence: {gesture_info['confidence']:.2f}"
            cv2.putText(canvas, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if 'frame_rate' in gesture_info:
            text = f"FPS: {gesture_info['frame_rate']:.1f}"
            cv2.putText(canvas, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

class AdvancedAnimationController:
    """
    Main animation controller that orchestrates all animation components
    """
    
    def __init__(self,
                 canvas_size: Tuple[int, int] = (800, 600),
                 smoothing_window: int = 10,
                 transition_frames: int = 15,
                 target_fps: int = 30):
        """
        Initialize advanced animation controller
        
        Args:
            canvas_size: Size of animation canvas
            smoothing_window: Size of motion smoothing window
            transition_frames: Number of frames for gesture transitions
            target_fps: Target frame rate for animations
        """
        self.canvas_size = canvas_size
        self.target_fps = target_fps
        
        # Initialize components
        self.motion_smoother = MotionSmoother(smoothing_window=smoothing_window)
        self.gesture_transition = GestureTransition(transition_frames=transition_frames)
        self.renderer = AnimationRenderer(canvas_size=canvas_size)
        
        # Animation state
        self.current_motion = None
        self.animation_queue = deque()
        self.last_frame_time = time.time()
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        
    def add_gesture_sequence(self, 
                           motion_sequence: np.ndarray, 
                           gesture_info: Dict) -> None:
        """
        Add a gesture sequence to the animation queue
        
        Args:
            motion_sequence: Motion sequence to animate (frames, landmarks, coordinates)
            gesture_info: Information about the gesture
        """
        # Smooth the motion sequence
        smoothed_sequence = self.motion_smoother.smooth_motion_sequence(motion_sequence)
        
        # Add to queue
        self.animation_queue.append({
            'motion': smoothed_sequence,
            'info': gesture_info,
            'frame_index': 0
        })
    
    def get_next_frame(self) -> np.ndarray:
        """
        Get the next animation frame
        
        Returns:
            Rendered animation frame
        """
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.frame_times.append(frame_time)
        self.last_frame_time = current_time
        
        # Calculate current FPS
        if len(self.frame_times) > 1:
            avg_frame_time = np.mean(list(self.frame_times))
            current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            current_fps = 0
        
        # Get current motion data
        motion_data = None
        gesture_info = {'frame_rate': current_fps}
        
        # Check for gesture transitions
        if self.gesture_transition.is_transitioning():
            motion_data = self.gesture_transition.get_transition_frame()
        
        # Process animation queue
        if motion_data is None and self.animation_queue:
            current_animation = self.animation_queue[0]
            motion_sequence = current_animation['motion']
            frame_idx = current_animation['frame_index']
            
            if frame_idx < len(motion_sequence):
                motion_data = motion_sequence[frame_idx]
                gesture_info.update(current_animation['info'])
                current_animation['frame_index'] += 1
            else:
                # Animation finished
                self.animation_queue.popleft()
                
                # Start transition to next animation if available
                if self.animation_queue:
                    next_animation = self.animation_queue[0]
                    if self.current_motion is not None:
                        self.gesture_transition.start_transition(
                            self.current_motion,
                            next_animation['motion'][0]
                        )
        
        # Update current motion
        if motion_data is not None:
            self.current_motion = motion_data
        
        # Render frame
        return self.renderer.render_frame(motion_data, gesture_info)
    
    def clear_queue(self) -> None:
        """Clear the animation queue"""
        self.animation_queue.clear()
    
    def is_animating(self) -> bool:
        """Check if currently animating"""
        return len(self.animation_queue) > 0 or self.gesture_transition.is_transitioning()
    
    def set_idle_animation(self, idle_motion: np.ndarray) -> None:
        """
        Set an idle animation that plays when no gestures are active
        
        Args:
            idle_motion: Idle motion sequence
        """
        if not self.is_animating():
            idle_info = {'name': 'Idle', 'confidence': 1.0}
            self.add_gesture_sequence(idle_motion, idle_info)

class MotionSynthesisIntegration:
    """
    Integration class that connects the research model with the animation controller
    """
    
    def __init__(self, 
                 model_path: str,
                 animation_controller: AdvancedAnimationController,
                 device: str = 'cpu'):
        """
        Initialize motion synthesis integration
        
        Args:
            model_path: Path to trained research model
            animation_controller: Animation controller instance
            device: Device for model inference
        """
        self.animation_controller = animation_controller
        self.device = torch.device(device)
        
        # Load research model
        self.model = self._load_research_model(model_path)
        self.model.eval()
        
        # Load class names
        self.class_names = self._load_class_names(model_path)
    
    def _load_research_model(self, model_path: str):
        """Load the research model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration
        config = checkpoint.get('config', {})
        
        # Import and create model
        from ..models.gesture_recognition.advanced_gesture_model import create_research_model
        model = create_research_model(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def _load_class_names(self, model_path: str) -> Dict[int, str]:
        """Load class names from model checkpoint"""
        checkpoint = torch.load(model_path, map_location='cpu')
        return checkpoint.get('class_names', {})
    
    def process_gesture_sequence(self, gesture_sequence: np.ndarray) -> None:
        """
        Process a gesture sequence and generate animation
        
        Args:
            gesture_sequence: Input gesture sequence (frames, features)
        """
        with torch.no_grad():
            # Convert to tensor
            input_tensor = torch.FloatTensor(gesture_sequence).unsqueeze(0).to(self.device)
            
            # Get prediction and synthesized motion
            results = self.model(input_tensor, generate_motion=True, motion_length=30)
            
            # Extract prediction
            predicted_class = torch.argmax(results['logits'], dim=1).item()
            confidence = torch.softmax(results['logits'], dim=1).max().item()
            
            # Get gesture name
            gesture_name = self.class_names.get(predicted_class, f"Gesture_{predicted_class}")
            
            # Extract synthesized motion
            motion_sequence = results['motion'][0].cpu().numpy()  # (frames, features)
            
            # Reshape to (frames, landmarks, coordinates)
            motion_sequence = motion_sequence.reshape(motion_sequence.shape[0], 21, 3)
            
            # Create gesture info
            gesture_info = {
                'name': gesture_name,
                'confidence': confidence,
                'predicted_class': predicted_class
            }
            
            # Add to animation controller
            self.animation_controller.add_gesture_sequence(motion_sequence, gesture_info)
    
    def predict_and_animate(self, gesture_sequence: np.ndarray) -> Dict:
        """
        Predict gesture and start animation
        
        Args:
            gesture_sequence: Input gesture sequence
            
        Returns:
            Prediction information
        """
        self.process_gesture_sequence(gesture_sequence)
        
        # Return prediction info
        with torch.no_grad():
            input_tensor = torch.FloatTensor(gesture_sequence).unsqueeze(0).to(self.device)
            results = self.model(input_tensor)
            
            predicted_class = torch.argmax(results['logits'], dim=1).item()
            confidence = torch.softmax(results['logits'], dim=1).max().item()
            gesture_name = self.class_names.get(predicted_class, f"Gesture_{predicted_class}")
            
            return {
                'gesture_name': gesture_name,
                'confidence': confidence,
                'predicted_class': predicted_class
            }