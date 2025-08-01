# 🌐 ASL Gesture Recognition Web App

A modern web interface for real-time ASL gesture recognition using Flask and WebSocket technology.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
cd project
python app.py
```

### 3. Open Your Browser
Navigate to: **http://localhost:5000**

## 📱 Features

### ✨ **Modern Web Interface**
- **Real-time camera feed** with WebSocket streaming
- **Live gesture recognition** with confidence scores
- **Recent predictions history** with timestamps
- **Responsive design** that works on desktop and mobile

### 🎯 **Recognition System**
- **68 trained gestures** from your ASL dataset
- **Real-time processing** at ~30 FPS
- **Confidence threshold** filtering (10% minimum)
- **Hand landmark detection** using MediaPipe

### 🎨 **UI Components**
- **Camera controls**: Start/Stop camera, camera selection
- **Live video feed** with hand landmark overlay
- **Current gesture display** with confidence bar
- **Recent predictions list** with clear functionality
- **Stats display**: Total gestures and current confidence

## 🔧 Technical Architecture

### **Backend (Flask)**
- **Flask web server** for HTTP endpoints
- **Flask-SocketIO** for real-time WebSocket communication
- **OpenCV** for camera capture and processing
- **PyTorch** for gesture recognition inference
- **MediaPipe** for hand landmark detection

### **Frontend (JavaScript)**
- **Socket.IO client** for real-time updates
- **Modern CSS** with gradients and animations
- **Responsive grid layout** for different screen sizes
- **Base64 video streaming** for cross-browser compatibility

## 📊 API Endpoints

### **REST Endpoints**
- `GET /` - Main web interface
- `POST /start_camera` - Start camera with specified ID
- `POST /stop_camera` - Stop camera feed
- `GET /get_predictions` - Get recent predictions and stats
- `POST /clear_predictions` - Clear prediction history

### **WebSocket Events**
- `connect` - Client connection established
- `disconnect` - Client disconnection
- `frame_update` - Real-time frame and prediction updates

## 🎮 Usage Instructions

### **Starting Camera**
1. Select your camera from the dropdown (usually Camera 0)
2. Click **"Start Camera"** button
3. Allow browser camera permissions if prompted
4. Camera feed will appear with "Camera Active" indicator

### **Gesture Recognition**
1. **Position your hand** clearly in front of the camera
2. **Hold steady** for best recognition (5-frame buffer)
3. **Watch confidence scores** - higher is better
4. **View predictions** in the right panel

### **Managing Predictions**
- **Recent predictions** show with timestamps
- **Click "Clear"** to reset prediction history
- **Confidence threshold** is set to 10% minimum

## 🔧 Configuration

### **Model Settings** (in `app.py`)
```python
# Recognition parameters
self.buffer_size = 5              # Frames for averaging
self.confidence_threshold = 0.1    # Minimum confidence (10%)
self.max_recent_predictions = 10   # Max stored predictions

# Camera settings
frame_width = 640                  # Video width
frame_height = 480                 # Video height
fps = 30                          # Frames per second
```

### **Camera Selection**
- **Camera 0**: Usually built-in webcam
- **Camera 1, 2, etc.**: External USB cameras
- **Auto-detection**: Try different values if camera doesn't work

## 🎨 Customization

### **Styling** (in `templates/index.html`)
- **Colors**: Modify CSS gradient and theme colors
- **Layout**: Adjust grid layout and component sizes
- **Animations**: Customize transitions and effects

### **Recognition Logic** (in `app.py`)
- **Confidence threshold**: Lower for more sensitive detection
- **Buffer size**: Increase for more stable recognition
- **Prediction filtering**: Modify duplicate detection logic

## 🐛 Troubleshooting

### **Camera Issues**
- **Permission denied**: Allow camera access in browser
- **Camera not found**: Try different camera IDs (0, 1, 2)
- **Black screen**: Check if camera is used by another app

### **Recognition Issues**
- **No predictions**: Ensure hands are visible and well-lit
- **Low confidence**: Improve lighting and hand positioning
- **Wrong predictions**: Model may need retraining with more data

### **Performance Issues**
- **Slow streaming**: Reduce frame rate or resolution
- **High CPU usage**: Close other applications
- **Memory leaks**: Restart app if running for extended periods

## 🆚 Web App vs Command Line

### **Web Application Advantages:**
- ✅ **Modern UI** with real-time updates
- ✅ **Multi-user access** through browser
- ✅ **Remote access** from any device on network
- ✅ **Prediction history** and statistics
- ✅ **Easy camera switching** without restart

### **Command Line Advantages:**
- ✅ **Lower latency** (direct OpenCV display)
- ✅ **Better performance** (no web encoding overhead)
- ✅ **Debugging output** in terminal
- ✅ **Simple deployment** (no web dependencies)

## 🌟 Next Steps

### **Enhancements**
- **Multi-camera support** for different angles
- **Gesture recording** for training new models
- **User accounts** with personal prediction history
- **Real-time model retraining** with user feedback
- **Mobile app** using the same Flask backend

### **Integration**
- **REST API** for other applications
- **Database storage** for prediction analytics
- **Cloud deployment** for public access
- **Authentication** for secure access

---

## 🎯 **Ready to Use!**

Your ASL gesture recognition system is now available as a modern web application! 

**Start the app and visit http://localhost:5000 to begin!** 🚀