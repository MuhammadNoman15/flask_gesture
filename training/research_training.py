#!/usr/bin/env python3
"""
Research-Grade ASL Gesture Recognition Training System

This advanced training system implements state-of-the-art techniques for
gesture recognition and motion synthesis research, including:
- Vision Transformers (ViT) for sequence modeling
- YOLOv8 integration for improved hand detection
- Support for research datasets (RWTH-PHOENIX-Weather, ASL Lexicon)
- Motion synthesis for generating fluid sign language animations
- Advanced deep learning techniques for gesture prediction
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
from typing import List, Tuple, Dict, Optional, Union
from tqdm import tqdm
from collections import defaultdict
# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Some visualizations will be skipped.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available. Some visualizations will be skipped.")

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import mediapipe as mp

# YOLOv8 integration
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLOv8 not available. Falling back to MediaPipe only.")

# Add project root to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.gesture_recognition.advanced_gesture_model import AdvancedGestureModel, create_research_model

class YOLOHandDetector:
    """
    YOLOv8-based hand detection for improved accuracy and speed
    """
    
    def __init__(self, model_path: Optional[str] = None):
        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8 not available. Install with: pip install ultralytics")
        
        # Load YOLOv8 model (can be pretrained or custom)
        if model_path and os.path.exists(model_path):
            self.yolo_model = YOLO(model_path)
        else:
            # Use pretrained YOLOv8 and fine-tune for hands
            self.yolo_model = YOLO('yolov8n.pt')  # Nano version for speed
        
        # MediaPipe as fallback for landmark extraction
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
    
    def detect_hands_yolo(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect hands using YOLOv8
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected hand regions
        """
        results = self.yolo_model(frame, classes=[0])  # Person class, can be customized for hands
        
        hand_regions = []
        for result in results:
            for box in result.boxes:
                if box.conf > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    hand_regions.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(box.conf)
                    })
        
        return hand_regions
    
    def extract_landmarks_from_region(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from detected region using MediaPipe
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Hand landmarks or None
        """
        x1, y1, x2, y2 = bbox
        hand_region = frame[y1:y2, x1:x2]
        
        if hand_region.size == 0:
            return None
        
        # Convert to RGB
        rgb_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_region)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                # Convert relative coordinates back to original frame coordinates
                abs_x = landmark.x * (x2 - x1) + x1
                abs_y = landmark.y * (y2 - y1) + y1
                landmarks.append([abs_x / frame.shape[1], abs_y / frame.shape[0], landmark.z])
            
            return self._extract_features(np.array(landmarks))
        
        return None
    
    def _extract_features(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract features from landmarks (same as original system)"""
        wrist = landmarks[0]
        features = []
        
        for i in range(21):
            if i < len(landmarks):
                point = landmarks[i]
                relative_x = point[0] - wrist[0]
                relative_y = point[1] - wrist[1]
                distance = np.sqrt(relative_x**2 + relative_y**2)
                features.append(distance)
            else:
                features.append(0.0)
        
        features = np.array(features, dtype=np.float32)
        if np.max(features) > 0:
            features = features / np.max(features)
        
        return features[:21]

class ResearchDatasetLoader:
    """
    Loader for research datasets including RWTH-PHOENIX-Weather and ASL Lexicon
    """
    
    def __init__(self, dataset_type: str, dataset_path: str):
        self.dataset_type = dataset_type.lower()
        self.dataset_path = dataset_path
        
        if self.dataset_type == 'rwth_phoenix':
            self.load_rwth_phoenix()
        elif self.dataset_type == 'asl_lexicon':
            self.load_asl_lexicon()
        elif self.dataset_type == 'custom_videos':
            self.load_custom_videos()
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def load_rwth_phoenix(self):
        """Load RWTH-PHOENIX-Weather dataset"""
        print("Loading RWTH-PHOENIX-Weather dataset...")
        
        # RWTH-PHOENIX dataset structure
        # annotations/manual/{train,dev,test}.corpus.csv
        # features/{train,dev,test}/*.png (frame images)
        
        self.data_splits = {}
        for split in ['train', 'dev', 'test']:
            corpus_file = os.path.join(self.dataset_path, 'annotations', 'manual', f'{split}.corpus.csv')
            if os.path.exists(corpus_file):
                self.data_splits[split] = self._parse_phoenix_corpus(corpus_file, split)
    
    def load_asl_lexicon(self):
        """Load ASL Lexicon dataset"""
        print("Loading ASL Lexicon dataset...")
        
        # ASL Lexicon structure varies, implement based on specific format
        # This is a template - adjust based on actual dataset structure
        
        video_dir = os.path.join(self.dataset_path, 'videos')
        annotation_file = os.path.join(self.dataset_path, 'annotations.json')
        
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            self.data_splits = self._process_asl_lexicon(annotations, video_dir)
    
    def load_custom_videos(self):
        """Load custom video dataset (like your current .webm files)"""
        print(f"Loading custom video dataset from {self.dataset_path}...")
        
        video_files = []
        labels = []
        
        for file in os.listdir(self.dataset_path):
            if file.endswith(('.webm', '.mp4', '.avi')):
                video_path = os.path.join(self.dataset_path, file)
                gesture_name = os.path.splitext(file)[0]
                
                video_files.append(video_path)
                labels.append(gesture_name)
        
        # Check class distribution
        from collections import Counter
        label_counts = Counter(labels)
        single_sample_classes = [label for label, count in label_counts.items() if count == 1]
        
        if single_sample_classes:
            print(f"⚠️ Found {len(single_sample_classes)} classes with only 1 sample. Using random split instead of stratified.")
            # Use random split for datasets with single-sample classes
            train_videos, test_videos, train_labels, test_labels = train_test_split(
                video_files, labels, test_size=0.2, random_state=42
            )
            
            train_videos, val_videos, train_labels, val_labels = train_test_split(
                train_videos, train_labels, test_size=0.2, random_state=42
            )
        else:
            # Use stratified split for balanced datasets
            train_videos, test_videos, train_labels, test_labels = train_test_split(
                video_files, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            train_videos, val_videos, train_labels, val_labels = train_test_split(
                train_videos, train_labels, test_size=0.2, random_state=42, stratify=train_labels
            )
        
        self.data_splits = {
            'train': list(zip(train_videos, train_labels)),
            'val': list(zip(val_videos, val_labels)),
            'test': list(zip(test_videos, test_labels))
        }
    
    def _parse_phoenix_corpus(self, corpus_file: str, split: str) -> List[Tuple]:
        """Parse RWTH-PHOENIX corpus file"""
        # Implementation depends on specific format
        # This is a template
        data = []
        # Parse corpus file and return list of (video_path, annotation) tuples
        return data
    
    def _process_asl_lexicon(self, annotations: Dict, video_dir: str) -> Dict:
        """Process ASL Lexicon annotations"""
        # Implementation depends on specific format
        # This is a template
        data_splits = {'train': [], 'val': [], 'test': []}
        # Process annotations and return splits
        return data_splits

class AdvancedGestureDataset(Dataset):
    """
    Advanced dataset class supporting multiple data formats and augmentations
    """
    
    def __init__(self, 
                 data_list: List[Tuple],
                 sequence_length: int = 16,  # Longer sequences for better context
                 augment: bool = False,
                 use_yolo: bool = True):
        """
        Initialize advanced gesture dataset
        
        Args:
            data_list: List of (video_path, label) tuples
            sequence_length: Number of frames per sequence
            augment: Whether to apply data augmentation
            use_yolo: Whether to use YOLOv8 for hand detection
        """
        self.data_list = data_list
        self.sequence_length = sequence_length
        self.augment = augment
        self.use_yolo = use_yolo
        
        # Initialize hand detector
        if use_yolo and YOLO_AVAILABLE:
            self.hand_detector = YOLOHandDetector()
        else:
            # Fallback to MediaPipe only
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
        
        # Create label mapping
        labels = [item[1] for item in data_list]
        unique_labels = sorted(list(set(labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        print(f"Dataset initialized with {len(unique_labels)} classes, {len(data_list)} samples")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        video_path, label = self.data_list[idx]
        label_idx = self.label_to_idx[label]
        
        # Extract features from video
        features = self._extract_video_features(video_path)
        
        if len(features) == 0:
            # Return zero sequence if extraction failed
            return torch.zeros(self.sequence_length, 21), label_idx
        
        # Create sequence
        sequence = self._create_sequence(features)
        
        return torch.FloatTensor(sequence), label_idx
    
    def _extract_video_features(self, video_path: str) -> List[np.ndarray]:
        """Extract features from video using advanced detection"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        features = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if self.use_yolo and hasattr(self, 'hand_detector'):
                # Use YOLOv8 + MediaPipe pipeline
                hand_regions = self.hand_detector.detect_hands_yolo(frame)
                
                if hand_regions:
                    # Use the most confident detection
                    best_region = max(hand_regions, key=lambda x: x['confidence'])
                    landmarks = self.hand_detector.extract_landmarks_from_region(
                        frame, best_region['bbox']
                    )
                    if landmarks is not None:
                        features.append(landmarks)
            else:
                # Fallback to MediaPipe only
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    landmarks = []
                    for landmark in results.multi_hand_landmarks[0].landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z])
                    
                    landmarks = np.array(landmarks)
                    features.append(self._extract_features(landmarks))
            
            frame_count += 1
        
        cap.release()
        return features
    
    def _extract_features(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract features from landmarks"""
        wrist = landmarks[0]
        features = []
        
        for i in range(21):
            if i < len(landmarks):
                point = landmarks[i]
                relative_x = point[0] - wrist[0]
                relative_y = point[1] - wrist[1]
                distance = np.sqrt(relative_x**2 + relative_y**2)
                features.append(distance)
            else:
                features.append(0.0)
        
        features = np.array(features, dtype=np.float32)
        if np.max(features) > 0:
            features = features / np.max(features)
        
        return features[:21]
    
    def _create_sequence(self, features: List[np.ndarray]) -> np.ndarray:
        """Create fixed-length sequence from variable-length features"""
        if len(features) == 0:
            return np.zeros((self.sequence_length, 21))
        
        # Interpolate or subsample to target length
        if len(features) < self.sequence_length:
            # Repeat features if too short
            while len(features) < self.sequence_length:
                features.extend(features[:min(len(features), self.sequence_length - len(features))])
        elif len(features) > self.sequence_length:
            # Subsample if too long
            indices = np.linspace(0, len(features) - 1, self.sequence_length).astype(int)
            features = [features[i] for i in indices]
        
        return np.array(features[:self.sequence_length])
    
    def get_class_names(self) -> Dict[int, str]:
        return self.idx_to_label
    
    def get_num_classes(self) -> int:
        return len(self.label_to_idx)

def train_research_model(model, train_loader, val_loader, config: Dict):
    """
    Train the research model with advanced techniques
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Advanced optimizer (AdamW with weight decay)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Advanced learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Loss functions
    classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    motion_loss = nn.MSELoss()
    
    # Training loop
    best_val_acc = 0
    writer = SummaryWriter(os.path.join(config['output_dir'], 'logs'))
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} - Training")
        
        for batch_idx, (data, labels) in enumerate(progress_bar):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with motion synthesis
            results = model(data, generate_motion=True, motion_length=30)
            
            # Classification loss
            cls_loss = classification_loss(results['logits'], labels)
            
            # Motion synthesis loss (self-supervised)
            if 'motion' in results:
                # Create target motion from input data (reconstruction)
                target_motion = data.unsqueeze(-1).repeat(1, 1, 1, 3).view(data.shape[0], data.shape[1], -1)
                target_motion = torch.nn.functional.pad(target_motion, (0, results['motion'].shape[-1] - target_motion.shape[-1]))
                motion_loss_val = motion_loss(results['motion'][:, :data.shape[1], :target_motion.shape[-1]], target_motion)
            else:
                motion_loss_val = 0
            
            # Combined loss
            total_loss = cls_loss + 0.1 * motion_loss_val
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            train_loss += total_loss.item()
            _, predicted = results['logits'].max(1)
            train_acc += predicted.eq(labels).sum().item()
            train_samples += labels.size(0)
            
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Acc': f'{100.*train_acc/train_samples:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_acc = 0
        val_samples = 0
        
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Validation"):
                data, labels = data.to(device), labels.to(device)
                
                results = model(data)
                loss = classification_loss(results['logits'], labels)
                
                val_loss += loss.item()
                _, predicted = results['logits'].max(1)
                val_acc += predicted.eq(labels).sum().item()
                val_samples += labels.size(0)
        
        # Update learning rate
        scheduler.step()
        
        # Calculate metrics
        train_acc_pct = 100. * train_acc / train_samples
        val_acc_pct = 100. * val_acc / val_samples
        
        # Logging
        writer.add_scalar('Loss/Train', train_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/Val', val_loss / len(val_loader), epoch)
        writer.add_scalar('Accuracy/Train', train_acc_pct, epoch)
        writer.add_scalar('Accuracy/Val', val_acc_pct, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch+1:3d}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc_pct:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_acc_pct:.2f}%")
        
        # Save best model
        if val_acc_pct > best_val_acc:
            best_val_acc = val_acc_pct
            
            # Save model with research metadata
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc_pct,
                'train_accuracy': train_acc_pct,
                'config': config,
                'model_type': 'AdvancedGestureModel',
                'features': 'ViT + Motion Synthesis + YOLOv8',
                'class_names': train_loader.dataset.get_class_names(),
                'timestamp': time.time()
            }
            
            torch.save(checkpoint, os.path.join(config['output_dir'], 'best_research_model.pth'))
            print(f"✅ New best model saved! Val Acc: {val_acc_pct:.2f}%")
    
    writer.close()
    return model, best_val_acc

def main():
    parser = argparse.ArgumentParser(description="Research-Grade ASL Gesture Training")
    parser.add_argument("--dataset_type", type=str, choices=['custom_videos', 'rwth_phoenix', 'asl_lexicon'],
                       default='custom_videos', help="Type of dataset to use")
    parser.add_argument("--dataset_path", type=str, default="dataset/ASL examples",
                       help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="models/gesture_recognition",
                       help="Output directory for models")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--sequence_length", type=int, default=16, help="Sequence length")
    parser.add_argument("--use_yolo", action="store_true", help="Use YOLOv8 for hand detection")
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-4,
        'sequence_length': args.sequence_length,
        'output_dir': args.output_dir,
        'use_yolo': args.use_yolo,
        
        # Model configuration
        'input_dim': 21,
        'd_model': 256,
        'nhead': 8,
        'num_transformer_layers': 6,
        'synthesis_hidden_dim': 512,
        'num_synthesis_layers': 4,
        'dropout': 0.1
    }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print("🎯 RESEARCH-GRADE ASL GESTURE RECOGNITION TRAINING")
    print("=" * 60)
    print(f"📊 Dataset Type: {args.dataset_type}")
    print(f"📁 Dataset Path: {args.dataset_path}")
    print(f"🔧 Use YOLOv8: {args.use_yolo}")
    print(f"🧠 Model: Vision Transformer + Motion Synthesis")
    print(f"💾 Output: {config['output_dir']}")
    print()
    
    # Load dataset
    dataset_loader = ResearchDatasetLoader(args.dataset_type, args.dataset_path)
    
    # Create datasets
    train_dataset = AdvancedGestureDataset(
        dataset_loader.data_splits['train'],
        sequence_length=config['sequence_length'],
        augment=True,
        use_yolo=config['use_yolo']
    )
    
    val_dataset = AdvancedGestureDataset(
        dataset_loader.data_splits.get('val', dataset_loader.data_splits.get('dev', [])),
        sequence_length=config['sequence_length'],
        augment=False,
        use_yolo=config['use_yolo']
    )
    
    # Update config with dataset info
    config['num_classes'] = train_dataset.get_num_classes()
    
    # Create data loaders (use num_workers=0 on Windows to avoid pickle errors)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=0, pin_memory=False)
    
    # Create research model
    print("🚀 Creating advanced research model...")
    model = create_research_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Train model
    print("\n🏋️ Starting research training...")
    trained_model, best_acc = train_research_model(model, train_loader, val_loader, config)
    
    print(f"\n✅ Training completed!")
    print(f"🎯 Best Validation Accuracy: {best_acc:.2f}%")
    print(f"📁 Model saved in: {config['output_dir']}")
    
    # Save final model info
    with open(os.path.join(config['output_dir'], 'research_model_info.json'), 'w') as f:
        json.dump({
            'model_type': 'AdvancedGestureModel',
            'features': ['Vision Transformer', 'Motion Synthesis', 'YOLOv8 Detection'],
            'best_accuracy': best_acc,
            'num_classes': config['num_classes'],
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': config
        }, f, indent=2)

if __name__ == "__main__":
    main()