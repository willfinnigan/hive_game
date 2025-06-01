from pathlib import Path
import os
import gc
import psutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime
import mlflow
import mlflow.pytorch

from hive.ml.data.dataset import HiveLazyGameDataset, collate_fn
from hive.trajectory.game_dataloader import GameDataLoader
from hive.ml.model.models import hive_gatv2

# fix for mac
import multiprocessing

multiprocessing.set_start_method('fork', force=True)

folder = Path(__file__).parents[3]

mlflow_dir = f"{folder}/mlruns"

print(mlflow_dir)
mlflow.set_tracking_uri(f"file://{mlflow_dir}")


def create_train_loader(dataset, batch_size, num_workers=2, prefetch_factor=2):
    """
    Create a DataLoader for the training dataset.

    Args:
        dataset: The HiveLazyGameDataset instance
        batch_size: Batch size for training
        num_workers: Number of worker threads for data loading

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=False,
        collate_fn=collate_fn
    )


def train(filepath, batch_size, model, device, optimizer, num_epochs=10,
          save_path=None, save_every=5, experiment_name="hive_training",
          start_epoch=0, train_history=None):
    """
    Train the model on the dataset with MLflow tracking.

    Args:
        filepath: Path to the dataset file
        batch_size: Batch size for training
        model: Model to train
        device: Device to train on
        optimizer: Optimizer to use
        num_epochs: Number of epochs to train for
        save_path: Path to save model checkpoints (optional)
        save_every: Save checkpoint every N epochs
        experiment_name: MLflow experiment name
        start_epoch: Epoch to start from (for resuming training)
        train_history: Dictionary containing training history (for resuming)
    """
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", optimizer.param_groups[0]['lr'])
        mlflow.log_param("weight_decay", optimizer.param_groups[0]['weight_decay'])
        mlflow.log_param("device", str(device))
        mlflow.log_param("model_type", model.__class__.__name__)
        mlflow.log_param("dataset_path", filepath)
        mlflow.log_param("save_every", save_every)

        # Log model architecture info if available
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param("total_parameters", total_params)
        mlflow.log_param("trainable_parameters", trainable_params)

        print(f"Training on device: {device}")
        print(f"Training for {num_epochs} epochs")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"MLflow experiment: {experiment_name}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

        start_time = time.time()

        # Move model to device
        model.to(device)

        # Create training dataset (only once)
        train_dataset = HiveLazyGameDataset(
            filepath,
            batch_size=batch_size,
        )
        train_loader = create_train_loader(
            train_dataset,
            batch_size=batch_size,
            num_workers=4,
            prefetch_factor=1
        )

        # Training metrics - initialize from history if provided
        train_losses = train_history['train_losses'] if train_history else []
        train_value_losses = train_history['train_value_losses'] if train_history else []
        train_value_accuracies = train_history['train_value_accuracies'] if train_history else []

        model.train()

        # Memory tracking variables
        memory_usage = []
        batch_times = []
        last_batch_time = time.time()

        # Training loop over epochs (starting from start_epoch)
        for epoch in range(start_epoch, start_epoch + num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_value_loss = 0.0
            value_accuracy = 0.0
            num_batches = 0

            # Log memory usage at start of epoch
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            print(f"\n[MEMORY] Epoch {epoch+1} start: RSS={mem_info.rss / (1024**2):.1f}MB, VMS={mem_info.vms / (1024**2):.1f}MB")
            
            # Force garbage collection before epoch
            collected = gc.collect()
            print(f"[GC] Collected {collected} objects before epoch {epoch+1}")
            
            # Check GPU memory if using CUDA or MPS
            if device.type == 'cuda':
                print(f"[CUDA] Allocated: {torch.cuda.memory_allocated(device) / (1024**2):.1f}MB")
                print(f"[CUDA] Cached: {torch.cuda.memory_reserved(device) / (1024**2):.1f}MB")
            elif device.type == 'mps':
                print(f"[MPS] Using Metal Performance Shaders on Mac")
                # Force MPS cache clear
                torch.mps.empty_cache()
                
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

                # Log batch metrics to MLflow (every 100 batches to avoid spam)
                if batch_idx % 5 == 0:
                    step = epoch * len(train_loader) + batch_idx
                    mlflow.log_metric("batch_loss", loss.item(), step=step)
                    mlflow.log_metric("batch_value_accuracy", value_acc, step=step)

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'value_acc': f"{value_acc:.4f}"
                })

                # Track batch processing time
                current_time = time.time()
                batch_time = current_time - last_batch_time
                batch_times.append(batch_time)
                last_batch_time = current_time
                
                # Log memory every 5 batches (more frequent for large datasets)
                if batch_idx % 5 == 0:
                    mem_info = process.memory_info()
                    memory_usage.append(mem_info.rss / (1024**2))  # RSS in MB
                    
                    # Calculate memory metrics
                    rss_mb = mem_info.rss / (1024**2)
                    vms_mb = mem_info.vms / (1024**2)
                    
                    # Log to MLFlow instead of just printing
                    step = epoch * len(train_loader) + batch_idx
                    mlflow.log_metric("memory_rss_mb", rss_mb, step=step)
                    mlflow.log_metric("memory_vms_mb", vms_mb, step=step)
                    mlflow.log_metric("batch_time", batch_time, step=step)

                    
                    # Check GPU memory if using CUDA or MPS
                    if device.type == 'cuda':
                        cuda_allocated = torch.cuda.memory_allocated(device) / (1024**2)
                        cuda_reserved = torch.cuda.memory_reserved(device) / (1024**2)
                        # Log CUDA memory to MLFlow
                        mlflow.log_metric("cuda_allocated_mb", cuda_allocated, step=step)
                        mlflow.log_metric("cuda_reserved_mb", cuda_reserved, step=step)
                        # Try to clear CUDA cache
                        torch.cuda.empty_cache()
                    elif device.type == 'mps':
                        # Log MPS usage (no direct metrics available)
                        mlflow.log_metric("device_type", 1, step=step)  # 1 for MPS
                        # Force MPS cache clear
                        torch.mps.empty_cache()
                
                # Clear memory
                del batch_data, outputs, value_preds, value_targets, loss, value_loss
                
                # More aggressive garbage collection
                gc.collect()
                
                # Clear MPS cache if using MPS
                if device.type == 'mps' and batch_idx % 20 == 0:
                    torch.mps.empty_cache()
                
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

            # Log memory usage at end of epoch
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / (1024**2)
            vms_mb = mem_info.vms / (1024**2)
            memory_change = rss_mb - memory_usage[0] if memory_usage else 0
            
            # Log to MLFlow
            mlflow.log_metric("epoch_end_rss_mb", rss_mb, step=epoch)
            mlflow.log_metric("epoch_end_vms_mb", vms_mb, step=epoch)
            mlflow.log_metric("epoch_memory_change_mb", memory_change, step=epoch)
            
            # Print summary
            print(f"\n[MEMORY] Epoch {epoch+1} end: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB")
            print(f"[MEMORY] Change during epoch: {memory_change:.1f}MB")
            
            # Check for memory growth trend
            if len(memory_usage) > 10:
                memory_growth = memory_usage[-1] - memory_usage[-10]
                mlflow.log_metric("memory_growth_last_10_mb", memory_growth, step=epoch)
                print(f"[MEMORY] Growth over last 10 measurements: {memory_growth:.1f}MB")
                
            # Check for batch time growth (indication of memory pressure)
            if len(batch_times) > 10:
                avg_first_10 = sum(batch_times[:10]) / 10
                avg_last_10 = sum(batch_times[-10:]) / 10
                time_ratio = avg_last_10/avg_first_10
                
                mlflow.log_metric("avg_batch_time_first_10", avg_first_10, step=epoch)
                mlflow.log_metric("avg_batch_time_last_10", avg_last_10, step=epoch)
                mlflow.log_metric("batch_time_ratio", time_ratio, step=epoch)
                
                print(f"[TIME] Avg batch time: first 10={avg_first_10:.3f}s, last 10={avg_last_10:.3f}s, ratio={time_ratio:.2f}x")
            
            # Force garbage collection after epoch
            collected = gc.collect()
            mlflow.log_metric("gc_collected_objects", collected, step=epoch)
            
            # Clean up DataLoader and workers
            del train_loader
            gc.collect()
            
            # Check GPU memory if using CUDA or MPS
            if device.type == 'cuda':
                cuda_allocated = torch.cuda.memory_allocated(device) / (1024**2)
                cuda_reserved = torch.cuda.memory_reserved(device) / (1024**2)
                
                mlflow.log_metric("epoch_end_cuda_allocated_mb", cuda_allocated, step=epoch)
                mlflow.log_metric("epoch_end_cuda_reserved_mb", cuda_reserved, step=epoch)
                
                print(f"[CUDA] Allocated: {cuda_allocated:.1f}MB")
                print(f"[CUDA] Cached: {cuda_reserved:.1f}MB")
                # Try to clear CUDA cache
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                print(f"[MPS] Clearing MPS cache at end of epoch")
                # Force MPS cache clear
                torch.mps.empty_cache()

            # Create new DataLoader for next epoch
            train_loader = create_train_loader(
                train_dataset,
                batch_size=batch_size,
                num_workers=4,
                prefetch_factor=1
            )

            # Log epoch metrics to MLflow
            mlflow.log_metric("epoch_loss", avg_epoch_loss, step=epoch)
            mlflow.log_metric("epoch_value_loss", avg_value_loss, step=epoch)
            mlflow.log_metric("epoch_value_accuracy", avg_value_accuracy, step=epoch)
            mlflow.log_metric("epoch_time", epoch_time, step=epoch)
            mlflow.log_metric("memory_usage_mb", mem_info.rss / (1024**2), step=epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s")
            print(f"Average Loss: {avg_epoch_loss:.4f}")
            print(f"Average Value Loss: {avg_value_loss:.4f}")
            print(f"Average Value Accuracy: {avg_value_accuracy:.4f}")

            # Save checkpoint
            if save_path and (epoch + 1) % save_every == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = f"{save_path}/hive_model_epoch_{epoch + 1}_{timestamp}.pt"

                checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_epoch_loss,
                    'train_losses': train_losses,
                    'train_value_losses': train_value_losses,
                    'train_value_accuracies': train_value_accuracies,
                }

                torch.save(checkpoint_data, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

                # Log checkpoint as MLflow artifact
                mlflow.log_artifact(checkpoint_path, f"checkpoints/epoch_{epoch + 1}")

        # Training complete
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")

        # Log final metrics
        mlflow.log_metric("total_training_time", total_time)
        mlflow.log_metric("final_loss", train_losses[-1] if train_losses else 0.0)
        mlflow.log_metric("final_value_loss", train_value_losses[-1] if train_value_losses else 0.0)
        mlflow.log_metric("final_value_accuracy", train_value_accuracies[-1] if train_value_accuracies else 0.0)

        # Save final model
        if save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"{save_path}/hive_model_final_{timestamp}.pt"

            final_model_data = {
                'epoch': num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses[-1] if train_losses else 0.0,
                'train_losses': train_losses,
                'train_value_losses': train_value_losses,
                'train_value_accuracies': train_value_accuracies,
            }

            torch.save(final_model_data, model_path)
            print(f"Final model saved to {model_path}")

            # Log final model as MLflow artifact
            mlflow.log_artifact(model_path, "final_model")

            # Log the model using MLflow's PyTorch integration
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="pytorch_model",
                registered_model_name=f"hive_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        # Create and log training curves plot
        try:
            import matplotlib.pyplot as plt

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

            epochs_range = range(1, len(train_losses) + 1)

            # Loss curve
            ax1.plot(epochs_range, train_losses, 'b-', label='Training Loss')
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)

            # Value loss curve
            ax2.plot(epochs_range, train_value_losses, 'r-', label='Value Loss')
            ax2.set_title('Value Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)

            # Value accuracy curve
            ax3.plot(epochs_range, train_value_accuracies, 'g-', label='Value Accuracy')
            ax3.set_title('Value Accuracy')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy')
            ax3.legend()
            ax3.grid(True)

            # Combined view
            ax4.plot(epochs_range, train_losses, 'b-', label='Total Loss', alpha=0.7)
            ax4_twin = ax4.twinx()
            ax4_twin.plot(epochs_range, train_value_accuracies, 'g-', label='Value Accuracy', alpha=0.7)
            ax4.set_title('Loss and Accuracy')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss', color='b')
            ax4_twin.set_ylabel('Accuracy', color='g')
            ax4.grid(True)

            plt.tight_layout()

            # Save and log the plot
            if save_path:
                plot_path = f"{save_path}/training_curves_{timestamp}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(plot_path, "plots")

            plt.close()

        except ImportError:
            print("Matplotlib not available - skipping training curves plot")
        except Exception as e:
            print(f"Error creating training curves plot: {e}")

        return {
            'train_losses': train_losses,
            'train_value_losses': train_value_losses,
            'train_value_accuracies': train_value_accuracies,
            'mlflow_run_id': mlflow.active_run().info.run_id
        }


if __name__ == "__main__":
    # Configuration variables - modify these to change training parameters
    # Path to checkpoint file to resume training from (set to None to start from scratch)
    CHECKPOINT_PATH = None  # Example: "hive/ml/supervised_training/checkpoints/20250101_120000/hive_model_epoch_10_20250101_120000.pt"
    
    # Training parameters
    TOTAL_EPOCHS = 50  # Total number of epochs to train for
    BATCH_SIZE = 128   # Batch size for training
    SAVE_EVERY = 1     # Save checkpoint every N epochs
    
    # Set up file paths and parameters
    folder = Path(__file__).parents[3]
    filepath = f"{folder}/game_strings/combined.txt"
    
    # Import memory monitoring
    import psutil
    process = psutil.Process(os.getpid())
    print(f"Initial memory usage: {process.memory_info().rss / (1024**2):.1f}MB")

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

    # Variables for checkpoint loading
    start_epoch = 0
    train_history = None
    
    # Check if resuming from checkpoint
    if CHECKPOINT_PATH:
        if os.path.exists(CHECKPOINT_PATH):
            print(f"Loading checkpoint from {CHECKPOINT_PATH}")
            
            # Move model to device before loading state dict
            model.to(device)
            
            # Load checkpoint with map_location to ensure tensors are loaded to the correct device
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            # First create a new optimizer with the model parameters on the correct device
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            
            # Then load the optimizer state
            # This is a workaround for the device mismatch issue
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            
            # Manually move optimizer state to the correct device
            for state in optimizer_state_dict['state'].values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            
            optimizer.load_state_dict(optimizer_state_dict)
            
            # Get the starting epoch (to continue from)
            start_epoch = checkpoint['epoch']
            
            # Load training history
            train_history = {
                'train_losses': checkpoint['train_losses'],
                'train_value_losses': checkpoint['train_value_losses'],
                'train_value_accuracies': checkpoint['train_value_accuracies']
            }
            
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Warning: Checkpoint file {CHECKPOINT_PATH} not found. Starting from scratch.")

    # Create save directory if it doesn't exist
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = checkpoint_dir / run_name
    os.makedirs(save_path, exist_ok=True)

    # Calculate remaining epochs
    remaining_epochs = TOTAL_EPOCHS - start_epoch
    if remaining_epochs <= 0:
        print(f"Warning: Starting epoch ({start_epoch}) >= total epochs ({TOTAL_EPOCHS})")
        print("Setting remaining epochs to 1")
        remaining_epochs = 1
    
    metrics = train(
        filepath=filepath,
        batch_size=BATCH_SIZE,
        model=model,
        device=device,
        optimizer=optimizer,
        num_epochs=remaining_epochs,
        save_path=save_path,
        save_every=SAVE_EVERY,
        experiment_name="hive_model_training",
        start_epoch=start_epoch,
        train_history=train_history
    )

    print(f"\nTraining completed!")
    print(f"MLflow run ID: {metrics['mlflow_run_id']}")
    print(f"View results in MLflow UI by running: mlflow ui")