import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import yaml
import os
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from models.gesture_recognition.gesture_model import GestureRecognitionModel

class SignLanguageDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 sequence_length: int = 30,
                 transform=None,
                 is_training: bool = True):
        """
        Initialize the sign language dataset.
        
        Args:
            data_path: Path to the dataset directory containing signdata.csv
            sequence_length: Length of gesture sequences
            transform: Optional transforms to apply
            is_training: Whether this is the training set
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.transform = transform
        self.is_training = is_training
        
        # Load and preprocess data
        self.data, self.labels, self.label_encoder = self._load_data()
        
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
        """Load and preprocess data from CSV files."""
        # Load main data
        data_df = pd.read_csv(os.path.join(self.data_path, 'signdata.csv'))
        
        # Select relevant features for gesture recognition
        # Using phonological features that are most relevant for gesture recognition
        feature_columns = [
            'Handshape.2.0', 'SelectedFingers.2.0', 'Flexion.2.0',
            'Spread.2.0', 'ThumbPosition.2.0', 'Movement.2.0',
            'MajorLocation.2.0', 'MinorLocation.2.0', 'Contact.2.0',
            'SignType.2.0', 'UlnarRotation.2.0'
        ]
        
        # Use EntryID as the label (unique sign identifier)
        label_column = 'EntryID'
        
        # Convert categorical features to numerical using one-hot encoding
        features = pd.get_dummies(data_df[feature_columns])
        
        # Encode labels
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(data_df[label_column])
        
        # Convert to numpy arrays
        features = features.values
        labels = labels.astype(np.int64)
        
        # Split into train/val sets
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Return appropriate split
        if self.is_training:
            return X_train, y_train, label_encoder
        else:
            return X_val, y_val, label_encoder
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a gesture sequence and its label.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple containing:
            - Gesture sequence tensor
            - Gesture label
        """
        # Get sequence data
        sequence_data = self.data[idx]
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence_data)
        
        if self.transform:
            sequence_tensor = self.transform(sequence_tensor)
        
        return sequence_tensor, self.labels[idx]
    
    def get_num_classes(self) -> int:
        """Get the number of unique gesture classes."""
        return len(self.label_encoder.classes_)
    
    def get_class_names(self) -> List[str]:
        """Get the list of gesture class names."""
        return self.label_encoder.classes_.tolist()

def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device,
                writer: SummaryWriter,
                epoch: int) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: Gesture recognition model
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        writer: TensorBoard writer
        epoch: Current epoch number
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (data, labels) in enumerate(progress_bar):
        # Move data to device
        data = data.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(data)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
        # Log to TensorBoard
        writer.add_scalar('train/batch_loss', loss.item(),
                         epoch * len(dataloader) + batch_idx)
    
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('train/epoch_loss', avg_loss, epoch)
    
    return avg_loss

def validate(model: nn.Module,
            dataloader: DataLoader,
            criterion: nn.Module,
            device: torch.device,
            writer: SummaryWriter,
            epoch: int) -> Tuple[float, float]:
    """
    Validate the model.
    
    Args:
        model: Gesture recognition model
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        writer: TensorBoard writer
        epoch: Current epoch number
        
    Returns:
        Tuple containing:
        - Average validation loss
        - Validation accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc="Validation"):
            # Move data to device
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, _ = model(data)
            loss = criterion(logits, labels)
            
            # Update statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    # Log to TensorBoard
    writer.add_scalar('val/loss', avg_loss, epoch)
    writer.add_scalar('val/accuracy', accuracy, epoch)
    
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description="Train Sign Language Gesture Recognition Model")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to training configuration file")
    parser.add_argument("--data_path", type=str, required=True,
                      help="Path to dataset directory containing signdata.csv")
    parser.add_argument("--output_dir", type=str, default="models/gesture_recognition",
                      help="Directory to save model checkpoints")
    parser.add_argument("--resume", type=str,
                      help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets and dataloaders
    train_dataset = SignLanguageDataset(
        args.data_path,
        sequence_length=config['sequence_length'],
        is_training=True
    )
    val_dataset = SignLanguageDataset(
        args.data_path,
        sequence_length=config['sequence_length'],
        is_training=False
    )
    
    # Update number of classes based on dataset
    config['num_classes'] = train_dataset.get_num_classes()
    
    # Create model with updated number of classes
    model = GestureRecognitionModel(
        input_size=train_dataset.data.shape[1],  # Use actual feature dimension
        hidden_size=config['hidden_size'],
        num_classes=config['num_classes'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Save class names for later use
    class_names = train_dataset.get_class_names()
    with open(os.path.join(args.output_dir, 'class_names.txt'), 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    # Set up training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=config['lr_patience'],
        verbose=True
    )
    
    # Set up TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # Resume training if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}")
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, config['epochs']):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, writer, epoch
        )
        
        # Validate
        val_loss, val_accuracy = validate(
            model, val_loader, criterion, device, writer, epoch
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }, os.path.join(args.output_dir, 'best_model.pth'))
        
        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }, os.path.join(args.output_dir, 'latest_model.pth'))
        
        print(f"Epoch {epoch}: "
              f"Train Loss = {train_loss:.4f}, "
              f"Val Loss = {val_loss:.4f}, "
              f"Val Accuracy = {val_accuracy:.4f}")
    
    writer.close()

if __name__ == "__main__":
    main() 