from pathlib import Path

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

# fix for mac
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

def train(filepath, batch_size, model, device, optimizer, num_epochs=10,
          save_path=None, save_every=5):
    """
    Train the model on the dataset.

    Args:
        filepath: Path to the dataset file
        batch_size: Batch size for training
        model: Model to train
        device: Device to train on
        optimizer: Optimizer to use
        num_epochs: Number of epochs to train for
        save_path: Path to save model checkpoints (optional)
        save_every: Save checkpoint every N epochs
    """
    print(f"Training on device: {device}")
    print(f"Training for {num_epochs} epochs")
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
    train_value_losses = []
    train_value_accuracies = []

    model.train()

    # Training loop over epochs
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_value_loss = 0.0
        value_accuracy = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None:
                continue

            # Move data to device
            batch_data = batch_data.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_data)

            # Calculate value prediction loss (MSE)
            value_preds = outputs["value"]
            value_targets = batch_data.value
            value_loss = F.mse_loss(value_preds, value_targets)

            # Calculate value prediction accuracy (sign match)
            value_acc = ((value_preds > 0) == (value_targets > 0)).float().mean().item()
            value_accuracy += value_acc

            # Total loss (currently just value loss)
            loss = value_loss

            # Backpropagation
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            epoch_value_loss += value_loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'value_loss': f"{value_loss.item():.4f}",
                'value_acc': f"{value_acc:.4f}"
            })

        # Calculate epoch averages
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            avg_value_loss = epoch_value_loss / num_batches
            avg_value_accuracy = value_accuracy / num_batches
        else:
            avg_epoch_loss = 0.0
            avg_value_loss = 0.0
            avg_value_accuracy = 0.0

        # Store epoch metrics
        train_losses.append(avg_epoch_loss)
        train_value_losses.append(avg_value_loss)
        train_value_accuracies.append(avg_value_accuracy)

        epoch_time = time.time() - epoch_start_time

        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s")
        print(f"Average Loss: {avg_epoch_loss:.4f}")
        print(f"Average Value Loss: {avg_value_loss:.4f}")
        print(f"Average Value Accuracy: {avg_value_accuracy:.4f}")

        # Save checkpoint
        if save_path and (epoch + 1) % save_every == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = f"{save_path}/hive_model_epoch_{epoch + 1}_{timestamp}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_epoch_loss,
                'train_losses': train_losses,
                'train_value_losses': train_value_losses,
                'train_value_accuracies': train_value_accuracies,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")

    # Save final model
    if save_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{save_path}/hive_model_final_{timestamp}.pt"
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses[-1] if train_losses else 0.0,
            'train_losses': train_losses,
            'train_value_losses': train_value_losses,
            'train_value_accuracies': train_value_accuracies,
        }, model_path)
        print(f"Final model saved to {model_path}")

    return {
        'train_losses': train_losses,
        'train_value_losses': train_value_losses,
        'train_value_accuracies': train_value_accuracies
    }


if __name__ == "__main__":
    # Set up file paths and parameters
    folder = Path(__file__).parents[3]
    filepath = f"{folder}/game_strings/combined.txt"
    batch_size = 64
    num_epochs = 20  # Set number of epochs

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

    this_dir = Path(__file__).parent
    checkpoint_dir = this_dir / "checkpoints"

    # Create save directory if it doesn't exist
    import os

    # create a run name based on datetime
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = checkpoint_dir / run_name
    os.makedirs(save_path, exist_ok=True)

    # Start training
    metrics = train(
        filepath=filepath,
        batch_size=batch_size,
        model=model,
        device=device,
        optimizer=optimizer,
        num_epochs=num_epochs,
        save_path=save_path,
        save_every=1
    )