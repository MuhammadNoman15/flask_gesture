#!/usr/bin/env python3
"""
ASL Gesture Recognition Training Script - Video-based Training

This script trains a gesture recognition model using video files from the ASL examples dataset.
Each video file represents one gesture class, and we extract hand landmarks to create training sequences.
"""

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
import time
import argparse
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from collections import defaultdict
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Optional visualization libraries
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available - plots will be skipped")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available - advanced plots will be skipped")

# Add the project root to the Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.gesture_recognition.gesture_model import GestureRecognitionModel

class VideoHandTracker:
    """Hand tracker for processing video files"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
    
    def extract_landmarks_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract hand landmarks from a single frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Use first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract landmark coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            
            landmarks = np.array(landmarks)
            
            # Extract features using the same method as hand_tracker.py
            return self._extract_features(landmarks)
        
        return None
    
    def _extract_features(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract features from landmarks (same as hand_tracker.py)"""
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
                distance = np.sqrt(relative_x**2 + relative_y**2)
                features.append(distance)
            else:
                features.append(0.0)
        
        # Normalize features to [0, 1] range
        features = np.array(features, dtype=np.float32)
        if np.max(features) > 0:
            features = features / np.max(features)
        
        # Ensure we have exactly 21 features
        return features[:21]


class ASLVideoDataset(Dataset):
    """Dataset for ASL video files"""
    
    def __init__(self, 
                 video_paths: List[str], 
                 labels: List[str], 
                 sequence_length: int = 8,  # Match main.py buffer size
                 augment: bool = False):
        """
        Initialize ASL video dataset
        
        Args:
            video_paths: List of paths to video files
            labels: List of gesture labels corresponding to videos
            sequence_length: Number of frames to use per sequence
            augment: Whether to apply data augmentation
        """
        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.augment = augment
        self.hand_tracker = VideoHandTracker()
        
        # Create label mapping
        unique_labels = sorted(list(set(labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        print(f"Dataset initialized with {len(unique_labels)} classes:")
        for label, idx in sorted(self.label_to_idx.items()):
            count = labels.count(label)
            print(f"  {idx:2d}: {label} ({count} videos)")
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]
        
        # Extract sequences from video
        sequences = self._extract_sequences_from_video(video_path)
        
        if len(sequences) == 0:
            # Return zero sequence if no hands detected
            return torch.zeros(self.sequence_length, 21), label_idx
        
        # Randomly select one sequence (for augmentation)
        if self.augment and len(sequences) > 1:
            sequence = sequences[np.random.randint(len(sequences))]
        else:
            # Use the middle sequence for consistency
            sequence = sequences[len(sequences) // 2]
        
        return torch.FloatTensor(sequence), label_idx
    
    def _extract_sequences_from_video(self, video_path: str) -> List[np.ndarray]:
        """Extract hand landmark sequences from video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return []
        
        frames_features = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract hand landmarks
            features = self.hand_tracker.extract_landmarks_from_frame(frame)
            if features is not None:
                frames_features.append(features)
            
            frame_count += 1
        
        cap.release()
        
        if len(frames_features) < self.sequence_length:
            if len(frames_features) == 0:
                return []
            # Repeat frames if video is too short
            while len(frames_features) < self.sequence_length:
                frames_features.extend(frames_features[:min(len(frames_features), 
                                                          self.sequence_length - len(frames_features))])
        
        # Create overlapping sequences
        sequences = []
        step_size = max(1, len(frames_features) // 4)  # Create multiple sequences per video
        
        for start_idx in range(0, len(frames_features) - self.sequence_length + 1, step_size):
            sequence = frames_features[start_idx:start_idx + self.sequence_length]
            sequences.append(np.array(sequence))
        
        return sequences
    
    def get_class_names(self) -> Dict[int, str]:
        """Get mapping from class index to class name"""
        return self.idx_to_label
    
    def get_num_classes(self) -> int:
        """Get number of classes"""
        return len(self.label_to_idx)


def load_video_dataset(dataset_path: str, sequence_length: int = 8, test_size: float = 0.2):
    """Load video dataset and split into train/validation"""
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Find all .webm video files
    video_files = []
    for file in os.listdir(dataset_path):
        if file.endswith('.webm'):
            video_files.append(file)
    
    if len(video_files) == 0:
        raise ValueError(f"No .webm files found in {dataset_path}")
    
    print(f"Found {len(video_files)} video files")
    
    # Create paths and labels
    video_paths = []
    labels = []
    
    for video_file in video_files:
        video_path = os.path.join(dataset_path, video_file)
        # Extract gesture name from filename (remove .webm extension)
        gesture_name = os.path.splitext(video_file)[0]
        
        video_paths.append(video_path)
        labels.append(gesture_name)
    
    # Split into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=test_size, random_state=42
        # Removed stratify=labels since each gesture has only 1 video
    )
    
    print(f"Training videos: {len(train_paths)}")
    print(f"Validation videos: {len(val_paths)}")
    
    # Create datasets
    train_dataset = ASLVideoDataset(train_paths, train_labels, sequence_length, augment=True)
    val_dataset = ASLVideoDataset(val_paths, val_labels, sequence_length, augment=False)
    
    return train_dataset, val_dataset


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} - Training")
    
    for batch_idx, (data, labels) in enumerate(progress_bar):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs, _ = model(data)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for data, labels in progress_bar:
            data, labels = data.to(device), labels.to(device)
            
            outputs, _ = model(data)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


def save_model_with_metadata(model, class_names, save_path, epoch, train_acc, val_acc):
    """Save model with metadata including class names"""
    
    # Create the checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': len(class_names),
        'input_size': 21,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'timestamp': time.time()
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")
    
    # Also save class names as JSON for easy loading
    class_names_path = os.path.splitext(save_path)[0] + '_class_names.json'
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to {class_names_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ASL Gesture Recognition from Videos")
    parser.add_argument("--dataset_path", type=str, 
                       default="dataset/ASL examples",
                       help="Path to directory containing .webm video files")
    parser.add_argument("--output_dir", type=str, 
                       default="models/gesture_recognition",
                       help="Directory to save trained models")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--sequence_length", type=int, default=8,
                       help="Sequence length for gesture recognition")
    parser.add_argument("--hidden_size", type=int, default=128,
                       help="Hidden size for LSTM")
    parser.add_argument("--num_layers", type=int, default=2,
                       help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout rate")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading video dataset...")
    train_dataset, val_dataset = load_video_dataset(
        args.dataset_path, 
        sequence_length=args.sequence_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0  # Fixed for Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0  # Fixed for Windows compatibility
    )
    
    # Create model
    num_classes = train_dataset.get_num_classes()
    print(f"Creating model with {num_classes} classes")
    
    model = GestureRecognitionModel(
        input_size=21,  # Hand landmark features
        hidden_size=args.hidden_size,
        num_classes=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Training loop
    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Print epoch results
        print(f"Epoch {epoch+1:2d}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.output_dir, 'best_visual_model.pth')
            save_model_with_metadata(
                model, train_dataset.get_class_names(), best_model_path,
                epoch, train_acc, val_acc
            )
            print(f"New best model saved! Validation accuracy: {val_acc:.2f}%")
        
        # Save latest model
        latest_model_path = os.path.join(args.output_dir, 'latest_visual_model.pth')
        save_model_with_metadata(
            model, train_dataset.get_class_names(), latest_model_path,
            epoch, train_acc, val_acc
        )
        
        print("-" * 60)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved in: {args.output_dir}")
    
    # Create final classification report
    print("\nGenerating final validation report...")
    model.eval()
    _, _, final_preds, final_labels = validate(model, val_loader, criterion, device)
    
    class_names = train_dataset.get_class_names()
    target_names = [class_names[i] for i in range(len(class_names))]
    
    print("\nClassification Report:")
    print(classification_report(final_labels, final_preds, target_names=target_names))


if __name__ == "__main__":
    main()