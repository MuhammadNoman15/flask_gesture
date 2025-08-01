# ASL Gesture Recognition Training Guide

This guide explains how to train your ASL gesture recognition model from video files.

## Overview

The training system processes video files (`.webm` format) from your dataset, extracts hand landmarks using MediaPipe, creates sequences of features, and trains a deep learning model for gesture recognition.

## Quick Start

### 1. Verify Dataset Structure

Make sure your dataset is organized like this:
```
project/
├── dataset/
│   └── ASL examples/
│       ├── ACCORDION.webm
│       ├── VIOLIN.webm
│       ├── SUIT.webm
│       ├── UNDERSTAND.webm
│       └── ... (more .webm files)
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Training (Simple)

```bash
python train_model.py
```

### 4. Run Training (Advanced)

```bash
python training/train_from_videos.py --dataset_path "dataset/ASL examples" --epochs 50 --batch_size 32
```

## Training Process

### What Happens During Training

1. **Video Processing**: Each `.webm` file is processed frame by frame
2. **Hand Detection**: MediaPipe detects hand landmarks in each frame
3. **Feature Extraction**: Hand landmarks are converted to 21-dimensional feature vectors
4. **Sequence Creation**: Features are grouped into sequences (default: 8 frames per sequence)
5. **Model Training**: LSTM-based model learns to classify gesture sequences
6. **Validation**: Model performance is evaluated on held-out data

### Model Architecture

- **Input**: 21 hand landmark features per frame
- **Sequence Length**: 8 frames (matches real-time buffer in main.py)
- **Model**: Bidirectional LSTM with attention mechanism
- **Hidden Size**: 128 units
- **Layers**: 2 LSTM layers
- **Output**: 86 gesture classes (based on your dataset)

## Training Parameters

### Basic Parameters

```bash
python training/train_from_videos.py \
    --dataset_path "dataset/ASL examples" \
    --output_dir "models/gesture_recognition" \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --sequence_length 8
```

### Advanced Parameters

- `--hidden_size 128`: LSTM hidden units
- `--num_layers 2`: Number of LSTM layers  
- `--dropout 0.2`: Dropout rate for regularization

## Output Files

After training, you'll find these files in `models/gesture_recognition/`:

1. **`best_visual_model.pth`**: Best performing model (use this for inference)
2. **`latest_visual_model.pth`**: Most recent model
3. **`best_visual_model_class_names.json`**: Class names mapping
4. **`latest_visual_model_class_names.json`**: Class names mapping

## Monitoring Training

The training script provides:
- Progress bars for each epoch
- Loss and accuracy metrics
- Automatic model saving
- Classification report at the end

## Troubleshooting

### Common Issues

1. **"No .webm files found"**
   - Check dataset path is correct
   - Ensure video files have `.webm` extension

2. **"CUDA out of memory"**
   - Reduce batch_size: `--batch_size 16`
   - Use CPU: Set CUDA_VISIBLE_DEVICES=""

3. **Low accuracy**
   - Increase epochs: `--epochs 100`
   - Adjust learning rate: `--learning_rate 0.0005`
   - Check video quality (clear hand visibility)

4. **Import errors**
   - Install missing packages: `pip install -r requirements.txt`
   - Check Python version (3.8+ recommended)

### Performance Tips

1. **GPU Training**: Ensure CUDA is available for faster training
2. **Data Quality**: Remove low-quality videos with unclear hand gestures
3. **Sequence Length**: Match training sequence length with inference buffer size
4. **Batch Size**: Increase if you have sufficient GPU memory

## Using Trained Model

After training, update your main application:

1. **Update model path** in `run_gesture_recognition.py`:
   ```python
   default_model_path = "models/gesture_recognition/best_visual_model.pth"
   ```

2. **Verify class count** in `main.py` matches your dataset:
   ```python
   num_classes=86,  # Should match number of .webm files
   ```

3. **Test the model**:
   ```bash
   python run_gesture_recognition.py
   ```

## Expected Results

With good quality videos and proper training:
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 75-90%  
- **Real-time Performance**: Should recognize gestures within 2-3 seconds

## Next Steps

1. **Train the model** using the provided scripts
2. **Test accuracy** with your trained model
3. **Update main.py** if needed based on your specific class count
4. **Fine-tune parameters** if accuracy is low

Good luck with your ASL gesture recognition training! 🚀