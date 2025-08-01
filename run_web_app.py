#!/usr/bin/env python3
"""
Easy startup script for ASL Gesture Recognition Web App
Checks dependencies and starts the Flask application
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask',
        'flask_socketio',
        'cv2',
        'torch',
        'mediapipe',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'flask_socketio':
                import flask_socketio
            else:
                __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    return missing_packages

def main():
    print("🚀 ASL Gesture Recognition Web App Startup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return 1
    
    print(f"✅ Python {sys.version.split()[0]} - OK")
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ app.py not found. Please run this script from the project directory.")
        return 1
    
    if not os.path.exists('templates/index.html'):
        print("❌ templates/index.html not found. Please ensure the web app files are present.")
        return 1
    
    print("✅ Project files - OK")
    
    # Check model file
    model_path = os.path.join('models', 'gesture_recognition', 'latest_visual_model.pth')
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure you have trained a model first using:")
        print("  python train_model.py")
        return 1
    
    print("✅ Model file - OK")
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Please install missing packages:")
        print("  pip install -r requirements.txt")
        return 1
    
    print("\n✅ All dependencies satisfied!")
    
    # Start the web app
    print("\n🌐 Starting web application...")
    print(f"📱 Open your browser to: http://localhost:5000")
    print("🔧 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Import and run the Flask app
        from app import socketio, app
        socketio.run(app, debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n⏹️ Server stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error starting web app: {e}")
        return 1

if __name__ == "__main__":
    exit(main())