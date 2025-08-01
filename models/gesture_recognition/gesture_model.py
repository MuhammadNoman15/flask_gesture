import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class GestureRecognitionModel(nn.Module):
    def __init__(self, 
                 input_size: int = 21,  # 21 hand landmarks features
                 hidden_size: int = 128,
                 num_classes: int = 86,  # Number of sign language gestures
                 num_layers: int = 2,
                 dropout: float = 0.2):
        """
        Initialize the gesture recognition model.
        
        Args:
            input_size: Size of input features (hand landmarks)
            hidden_size: Size of hidden layers
            num_classes: Number of gesture classes to recognize
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
        """
        super(GestureRecognitionModel, self).__init__()
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, 
                x: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            lengths: Optional tensor of sequence lengths
            
        Returns:
            Tuple containing:
            - Logits for gesture classification
            - Attention weights
        """
        # Pack sequence if lengths are provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Unpack sequence if it was packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Calculate attention weights
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        logits = self.classifier(context)
        
        return logits, attention_weights
    
    def predict(self, 
                x: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Make predictions for input sequences.
        
        Args:
            x: Input tensor
            lengths: Optional sequence lengths
            
        Returns:
            Predicted gesture class indices
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x, lengths)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def save_model(self, path: str):
        """Save model state dictionary."""
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load_model(cls, 
                  path: str, 
                  input_size: int = 21,
                  hidden_size: int = 128,
                  num_classes: int = 86,
                  num_layers: int = 2,
                  dropout: float = 0.2) -> 'GestureRecognitionModel':
        """
        Load a saved model.
        
        Args:
            path: Path to saved model state dictionary
            input_size: Input feature size
            hidden_size: Hidden layer size
            num_classes: Number of gesture classes
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            
        Returns:
            Loaded model instance
        """
        model = cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout
        )
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        return model