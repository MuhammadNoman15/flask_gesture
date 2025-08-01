#!/usr/bin/env python3
"""
Simple ASL Gesture Recognition System

Usage:
    python run_gesture_recognition.py
    python run_gesture_recognition.py --model_path path/to/model.pth
"""

import os
import sys

def main():
    # Add the project src directory to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, 'src')
    sys.path.insert(0, src_path)
    
    # Use latest_visual_model.pth as default
    default_model_path = os.path.join(project_root, 'models', 'gesture_recognition', 'latest_model.pth')
    
    # Simple argument handling
    model_path = default_model_path
    if len(sys.argv) > 1 and sys.argv[1] == '--model_path' and len(sys.argv) > 2:
        model_path = sys.argv[2]
    
    # Verify model exists
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found at {model_path}")
        return 1
    
    print("🎯 ASL GESTURE RECOGNITION SYSTEM")
    print("=" * 40)
    print(f"📁 Model: {os.path.basename(model_path)}")
    print("📹 Camera: 0 (default)")
    print("🔧 Mode: Simple & Accurate")
    print()
    
    # Import and run the system
    try:
        from src.main import SignLanguageAvatar
        
        print("🚀 Starting system...")
        
        # Create simple system
        avatar = SignLanguageAvatar(
            model_path=model_path,
            camera_id=0,
            fps=30,
            enable_debug=True
        )
        
        avatar.run()
        
    except KeyboardInterrupt:
        print("\n⏹️ System stopped by user")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure dependencies are installed:")
        print("pip install opencv-python mediapipe torch numpy")
        return 1
    except Exception as e:
        print(f"❌ System error: {e}")
        return 1
    
    print("👋 Thanks for using ASL Gesture Recognition!")
    return 0

if __name__ == "__main__":
    exit(main())