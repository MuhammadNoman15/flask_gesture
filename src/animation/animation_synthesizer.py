import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
from scipy.interpolate import CubicSpline

class AnimationSynthesizer:
    def __init__(self, 
                 fps: int = 30,
                 smoothing_window: int = 5,
                 interpolation_method: str = 'cubic'):
        """
        Initialize the animation synthesizer.
        
        Args:
            fps: Frames per second for the animation
            smoothing_window: Window size for smoothing motion
            interpolation_method: Method for interpolating between keyframes
        """
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.interpolation_method = interpolation_method
        self.current_sequence = []
        self.target_sequence = []
        
    def add_gesture_sequence(self, 
                           gesture_sequence: List[Dict],
                           transition_frames: int = 15) -> None:
        """
        Add a sequence of gestures to be animated.
        
        Args:
            gesture_sequence: List of gesture dictionaries containing pose data
            transition_frames: Number of frames to use for transitions
        """
        if not self.current_sequence:
            self.current_sequence = gesture_sequence
            self.target_sequence = gesture_sequence
        else:
            # Add transition frames between current and new sequence
            transition = self._create_transition(
                self.current_sequence[-1],
                gesture_sequence[0],
                transition_frames
            )
            self.current_sequence.extend(transition)
            self.current_sequence.extend(gesture_sequence[1:])
            self.target_sequence = gesture_sequence
    
    def _create_transition(self, 
                          start_pose: Dict,
                          end_pose: Dict,
                          num_frames: int) -> List[Dict]:
        """
        Create a smooth transition between two poses.
        
        Args:
            start_pose: Starting pose dictionary
            end_pose: Ending pose dictionary
            num_frames: Number of frames for the transition
            
        Returns:
            List of interpolated poses
        """
        transition = []
        
        # Extract landmark coordinates
        start_points = np.array([[lm['x'], lm['y'], lm['z']] 
                               for lm in start_pose['landmarks']])
        end_points = np.array([[lm['x'], lm['y'], lm['z']] 
                             for lm in end_pose['landmarks']])
        
        # Create interpolation for each landmark
        for i in range(len(start_points)):
            if self.interpolation_method == 'cubic':
                # Create cubic spline interpolation
                x_spline = CubicSpline([0, num_frames-1], 
                                     [start_points[i, 0], end_points[i, 0]])
                y_spline = CubicSpline([0, num_frames-1], 
                                     [start_points[i, 1], end_points[i, 1]])
                z_spline = CubicSpline([0, num_frames-1], 
                                     [start_points[i, 2], end_points[i, 2]])
                
                # Generate interpolated points
                frames = np.arange(num_frames)
                x_interp = x_spline(frames)
                y_interp = y_spline(frames)
                z_interp = z_spline(frames)
                
                # Add interpolated points to transition
                for j in range(num_frames):
                    if j >= len(transition):
                        transition.append({
                            'landmarks': [],
                            'handedness': start_pose['handedness'],
                            'confidence': start_pose['confidence']
                        })
                    transition[j]['landmarks'].append({
                        'x': float(x_interp[j]),
                        'y': float(y_interp[j]),
                        'z': float(z_interp[j])
                    })
        
        return transition
    
    def smooth_motion(self, sequence: List[Dict]) -> List[Dict]:
        """
        Apply motion smoothing to reduce jitter.
        
        Args:
            sequence: List of pose dictionaries
            
        Returns:
            Smoothed sequence of poses
        """
        if len(sequence) < self.smoothing_window:
            return sequence
            
        smoothed = []
        window = self.smoothing_window
        
        for i in range(len(sequence)):
            # Get window of poses
            start_idx = max(0, i - window // 2)
            end_idx = min(len(sequence), i + window // 2 + 1)
            window_poses = sequence[start_idx:end_idx]
            
            # Average landmark positions
            avg_landmarks = []
            for j in range(len(sequence[i]['landmarks'])):
                x = np.mean([p['landmarks'][j]['x'] for p in window_poses])
                y = np.mean([p['landmarks'][j]['y'] for p in window_poses])
                z = np.mean([p['landmarks'][j]['z'] for p in window_poses])
                
                avg_landmarks.append({
                    'x': float(x),
                    'y': float(y),
                    'z': float(z)
                })
            
            smoothed.append({
                'landmarks': avg_landmarks,
                'handedness': sequence[i]['handedness'],
                'confidence': sequence[i]['confidence']
            })
        
        return smoothed
    
    def get_next_frame(self) -> Optional[Dict]:
        """
        Get the next frame in the animation sequence.
        
        Returns:
            Dictionary containing pose data for the next frame, or None if sequence is complete
        """
        if not self.current_sequence:
            return None
            
        # Get and remove the next frame
        frame = self.current_sequence.pop(0)
        
        # Apply smoothing if needed
        if len(self.current_sequence) >= self.smoothing_window:
            self.current_sequence = self.smooth_motion(self.current_sequence)
        
        return frame
    
    def reset(self):
        """Reset the animation sequence."""
        self.current_sequence = []
        self.target_sequence = []
    
    def get_remaining_frames(self) -> int:
        """Get the number of remaining frames in the sequence."""
        return len(self.current_sequence)
    
    def is_complete(self) -> bool:
        """Check if the animation sequence is complete."""
        return len(self.current_sequence) == 0 