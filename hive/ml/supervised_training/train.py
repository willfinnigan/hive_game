import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime

from hive.ml.data.dataset import HiveLazyGameDataset, collate_fn
from hive.trajectory.game_dataloader import GameDataLoader
from hive.ml.model.models import hive_gatv2



def train(filepath, batch_size, model, device, optimizer,
          save_path=None,
          move_loss_weight=1.0, value_loss_weight=0.5):
    """
    Train the model on the dataset.
    
    Args:
        filepath: Path to the dataset file
        batch_size: Batch size for training
        model: Model to train
        device: Device to train on
        optimizer: Optimizer to use
        save_path: Path to save model checkpoints (optional)
        move_loss_weight: Weight for move prediction loss
        value_loss_weight: Weight for value prediction loss
    """
    print(f"Training on device: {device}")
    start_time = time.time()
    
    # Move model to device
    model.to(device)
    
    # Create training dataset and loader
    train_dataset = HiveLazyGameDataset(filepath, batch_size=batch_size)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        prefetch_factor=3,
        collate_fn=collate_fn)
    
    # Training metrics
    train_losses = []
    
    model.train()

    epoch_loss = 0.0
    epoch_move_loss = 0.0
    epoch_value_loss = 0.0
    move_accuracy = 0.0
    value_accuracy = 0.0
    num_batches = 0
        
    progress_bar = tqdm(train_loader, desc="Training")
        
    for batch_idx, batch_data in enumerate(progress_bar):
        if batch_data is None:
            continue
                
        # Move data to device
        batch_data = batch_data.to(device)
            
        # Zero gradients
        optimizer.zero_grad()
            
        # Forward pass
        outputs = model(batch_data)
            
        # Calculate move prediction loss (binary cross entropy)
        move_logits = outputs["move_predictor"]
        move_labels = batch_data.move_labels
            
        # Check if we have valid moves
        if len(move_logits) > 0 and len(move_labels) > 0:
            move_loss = F.binary_cross_entropy_with_logits(move_logits, move_labels)
                
            # Calculate move prediction accuracy
            move_preds = (torch.sigmoid(move_logits) > 0.5).float()
            move_acc = (move_preds == move_labels).float().mean().item()
            move_accuracy += move_acc
        else:
            move_loss = torch.tensor(0.0, device=device)
                
        # Calculate value prediction loss (MSE)
        value_preds = outputs["value_predictor"]
        value_targets = batch_data.winner
        value_loss = F.mse_loss(value_preds, value_targets)
            
        # Calculate value prediction accuracy (sign match)
        value_acc = ((value_preds > 0) == (value_targets > 0)).float().mean().item()
        value_accuracy += value_acc
            
        # Combine losses with weights
        loss = move_loss_weight * move_loss + value_loss_weight * value_loss
            
        # Backpropagation
        loss.backward()
            
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
        # Optimizer step
        optimizer.step()
            
        # Update metrics
        epoch_loss += loss.item()
        epoch_move_loss += move_loss.item()
        epoch_value_loss += value_loss.item()
        num_batches += 1
            
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'move_loss': f"{move_loss.item():.4f}",
            'value_loss': f"{value_loss.item():.4f}",
            'move_acc': f"{move_acc:.4f}" if 'move_acc' in locals() else "N/A",
            'value_acc': f"{value_acc:.4f}"
        })
        
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    
    # Save final model if no validation was done
    if save_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{save_path}/hive_model_final_{timestamp}.pt"
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses[-1],
        }, model_path)
        print(f"Final model saved to {model_path}")


if __name__ == "__main__":
    # Set up file paths and parameters
    filepath = "game_strings/combined.txt"
    batch_size = 32
    save_path = "models"  # Directory to save model checkpoints
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Get model
    model = hive_gatv2

    # Set up optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Create save directory if it doesn't exist
    import os
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Start training
    train(
        filepath=filepath,
        batch_size=batch_size,
        model=model,
        device=device,
        optimizer=optimizer,
        save_path=save_path,
        move_loss_weight=1.0,
        value_loss_weight=0.5
    )

