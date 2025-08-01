#!/usr/bin/env python3
"""
Simple script to train ASL gesture recognition model from video files.

Usage:
    python train_model.py
    python train_model.py --epochs 30 --batch_size 16
"""

import os
import sys

def main():
    # Add project root to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Default paths (relative to project root)
    dataset_path = os.path.join(project_root, "dataset", "ASL examples")
    output_dir = os.path.join(project_root, "models", "gesture_recognition")
    
    # Verify dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: Dataset not found at {dataset_path}")
        print("Please ensure your ASL video files are in the correct location.")
        return 1
    
    # Count video files
    video_files = [f for f in os.listdir(dataset_path) if f.endswith('.webm')]
    print(f"🎯 ASL GESTURE RECOGNITION TRAINING")
    print("=" * 50)
    print(f"📁 Dataset: {dataset_path}")
    print(f"📊 Videos found: {len(video_files)}")
    print(f"💾 Output: {output_dir}")
    print()
    
    # Import and run training
    try:
        from training.train_from_videos import main as train_main
        
        # Set up arguments (you can modify these)
        sys.argv = [
            'train_from_videos.py',
            '--dataset_path', dataset_path,
            '--output_dir', output_dir,
            '--epochs', '100',
            '--batch_size', '1',
            '--learning_rate', '0.0005',
            '--sequence_length', '4'
        ]
        
        print("🚀 Starting training...")
        train_main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install torch torchvision opencv-python mediapipe numpy tqdm scikit-learn matplotlib seaborn tensorboard")
        return 1
    except Exception as e:
        print(f"❌ Training error: {e}")
        return 1
    
    print("\n✅ Training completed successfully!")
    print(f"📁 Check {output_dir} for trained models")
    return 0

if __name__ == "__main__":
    exit(main())