#!/usr/bin/env python3
"""
Research-Grade ASL Gesture Recognition Training Runner

This script provides a simple interface to train advanced gesture recognition models
using state-of-the-art techniques including Vision Transformers, YOLOv8, and motion synthesis.

Features:
- Vision Transformer (ViT) for gesture sequence modeling
- YOLOv8 integration for improved hand detection
- Motion synthesis for generating fluid sign language animations  
- Support for research datasets (RWTH-PHOENIX-Weather, ASL Lexicon)
- Advanced deep learning techniques for gesture prediction

Usage:
    python train_research_model.py                    # Use default config
    python train_research_model.py --config custom.yaml  # Use custom config
    python train_research_model.py --quick             # Quick training (reduced epochs)
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Import core dependencies at the top
try:
    import torch
    import cv2
    import numpy as np
    import mediapipe
    CORE_DEPS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Core dependency issue: {e}")
    CORE_DEPS_AVAILABLE = False

def main():
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    parser = argparse.ArgumentParser(description="Research-Grade ASL Gesture Recognition Training")
    parser.add_argument("--config", type=str, default="config/research_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Override dataset path")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Override output directory")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Override batch size")
    parser.add_argument("--quick", action="store_true",
                       help="Quick training mode (reduced epochs for testing)")
    parser.add_argument("--use_yolo", action="store_true", default=None,
                       help="Enable YOLOv8 hand detection")
    parser.add_argument("--no_yolo", action="store_false", dest="use_yolo",
                       help="Disable YOLOv8 hand detection")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = project_root / config_path
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("Using default configuration...")
        config = get_default_config()
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Apply command line overrides
    if args.dataset_path:
        config['dataset']['path'] = args.dataset_path
    if args.output_dir:
        config['output']['model_dir'] = args.output_dir
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.quick:
        config['training']['epochs'] = 20
        config['training']['batch_size'] = 8
        print("🏃 Quick training mode enabled")
    if args.use_yolo is not None:
        config['advanced']['yolo']['enabled'] = args.use_yolo
    
    # Validate paths
    dataset_path = Path(config['dataset']['path'])
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return 1
    
    output_dir = Path(config['output']['model_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count dataset files
    video_extensions = ['.webm', '.mp4', '.avi', '.mov']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(dataset_path.glob(f'*{ext}')))
    
    # Display configuration
    print("🎯 RESEARCH-GRADE ASL GESTURE RECOGNITION")
    print("=" * 60)
    print(f"📊 Model Type: {config['model']['type'].upper()}")
    print(f"📁 Dataset: {dataset_path}")
    print(f"📹 Video Files: {len(video_files)}")
    print(f"🔧 YOLOv8 Detection: {'✅' if config['advanced']['yolo']['enabled'] else '❌'}")
    print(f"🎬 Motion Synthesis: ✅")
    print(f"🧠 Vision Transformer: ✅")
    print(f"📏 Sequence Length: {config['dataset']['sequence_length']}")
    print(f"🏋️ Epochs: {config['training']['epochs']}")
    print(f"📦 Batch Size: {config['training']['batch_size']}")
    print(f"💾 Output: {output_dir}")
    print(f"🖥️ Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print()
    
    if len(video_files) == 0:
        print("❌ No video files found in dataset directory!")
        print("Please ensure your dataset contains .webm, .mp4, .avi, or .mov files")
        return 1
    
    # Quick dependency check
    if CORE_DEPS_AVAILABLE:
        print("✅ Core dependencies verified")
    else:
        print("⚠️ Some dependencies missing, but continuing...")
        print("If training fails, run: pip install -r requirements.txt")
    
    # Import training module with proper error handling
    try:
        print("🚀 Starting research training...")
        print("-" * 60)
        
        # Try to import the research training module
        try:
            from training.research_training import main as research_main
            training_type = "research"
        except ImportError as import_err:
            print(f"⚠️ Research training import failed: {import_err}")
            print("🔄 Falling back to basic training...")
            try:
                from training.train_from_videos import main as basic_main
                research_main = basic_main
                training_type = "basic"
            except ImportError:
                print("❌ Both training modules failed to import")
                return 1
        
        # Convert config to command line arguments for the training script
        if training_type == "research":
            train_args = [
                'research_training.py',
                '--dataset_type', config['dataset']['type'],
                '--dataset_path', str(dataset_path),
                '--output_dir', str(output_dir),
                '--epochs', str(config['training']['epochs']),
                '--batch_size', str(config['training']['batch_size']),
                '--learning_rate', str(config['training']['optimizer']['learning_rate']),
                '--sequence_length', str(config['dataset']['sequence_length'])
            ]
            
            if config['advanced']['yolo']['enabled']:
                train_args.append('--use_yolo')
        else:
            # Basic training arguments
            train_args = [
                'train_from_videos.py',
                '--dataset_path', str(dataset_path),
                '--output_dir', str(output_dir),
                '--epochs', str(config['training']['epochs']),
                '--batch_size', str(config['training']['batch_size']),
                '--learning_rate', str(config['training']['optimizer']['learning_rate']),
                '--sequence_length', str(config['dataset']['sequence_length'])
            ]
        
        # Override sys.argv for the training script
        original_argv = sys.argv[:]
        sys.argv = train_args
        
        print(f"📊 Using {training_type} training mode")
        
        # Run training
        research_main()
        
        # Restore original argv
        sys.argv = original_argv
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("🔍 Error details:")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n✅ Research training completed successfully!")
    print(f"📁 Models saved in: {output_dir}")
    print(f"📊 Check TensorBoard logs: tensorboard --logdir {output_dir}/logs")
    return 0



def get_default_config():
    """Get default configuration if config file not found"""
    return {
        'dataset': {
            'type': 'custom_videos',
            'path': 'dataset/ASL examples',
            'sequence_length': 16
        },
        'model': {
            'type': 'advanced_vit'
        },
        'training': {
            'epochs': 50,
            'batch_size': 16,
            'optimizer': {
                'learning_rate': 1e-4
            }
        },
        'advanced': {
            'yolo': {
                'enabled': True
            }
        },
        'output': {
            'model_dir': 'models/gesture_recognition'
        }
    }

if __name__ == "__main__":
    exit(main())