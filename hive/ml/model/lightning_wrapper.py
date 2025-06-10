import lightning as L
import torch
import torch.nn.functional as F
from torch import optim
import time
from torch_geometric.utils import to_dense_batch

class HiveLightningModel(L.LightningModule):

    def __init__(self,
                 model,
                 learning_rate=0.01,
                 weight_decay=5e-4,
                 policy_weight=1,
                 value_weight=0.2,
                 min_step_for_value=6):
        
        super().__init__()
        # Save hyperparameters but ignore the model to avoid the warning
        self.save_hyperparameters(ignore=['model'])
        self.model = model

    def forward(self, batch):
        return self.model(batch)

    def _calculate_weighted_value_loss(self, value_preds, value_targets, batch, phase):
        """Calculate weighted value loss with per-sample weighting and log metrics"""
        # Input validation
        if not hasattr(batch, 'value'):
            raise ValueError("Batch must have 'value' attribute")
        if not hasattr(batch, 'step'):
            raise ValueError("Batch must have 'step' attribute")
        
        batch_size = value_preds.size(0)

        if value_targets.size(0) != batch_size:
            raise ValueError(f"Value predictions ({value_preds.size(0)}) and targets ({value_targets.size(0)}) have different batch sizes")
        
        # Validate tensor dimensions
        if value_preds.dim() != 1:
            raise ValueError(f"Value predictions must be 1D tensor, got {value_preds.dim()}D")
        if value_targets.dim() != 1:
            raise ValueError(f"Value targets must be 1D tensor, got {value_targets.dim()}D")
        
        # Create per-sample weights for value loss
        # Only apply value loss to samples where step >= min_step_for_value
        # AND value != 0 (as 0 may represent a draw, which we don't train the value head on)
        step_mask = batch.step >= self.hparams.min_step_for_value
        value_mask = batch.value != 0
        sample_weights = (step_mask & value_mask).float()
        
        # Calculate per-sample value loss and apply weighting
        value_loss_per_sample = F.mse_loss(value_preds, value_targets, reduction='none')
        
        # Check for NaN values and handle gracefully
        if torch.isnan(value_loss_per_sample).any() or torch.isnan(value_preds).any():
            nan_indicator = torch.tensor(1.0, device=value_preds.device, dtype=value_preds.dtype)
            self.log(f'{phase}_nan_detected', nan_indicator, on_step=(phase=='train'), on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
            value_loss_per_sample = torch.nan_to_num(value_loss_per_sample, nan=0.0)
            value_preds = torch.nan_to_num(value_preds, nan=0.0)
        
        # Proper handling of zero weights to avoid division by zero
        num_weighted_samples = sample_weights.sum()
        if num_weighted_samples > 0:
            weighted_value_loss = (value_loss_per_sample * sample_weights).sum() / num_weighted_samples
        else:
            weighted_value_loss = value_preds.sum() * 0.0 # Zero loss with proper gradient flow
        
        # Validate loss is finite after computation and provide a fallback
        if not torch.isfinite(weighted_value_loss):
            fallback_indicator = torch.tensor(1.0, device=value_preds.device, dtype=value_preds.dtype)
            self.log(f'{phase}_value_loss_fallback', fallback_indicator, on_step=(phase=='train'), on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
            weighted_value_loss = value_preds.sum() * 0.0 # Zero loss with proper gradient flow
        
        # Calculate value accuracy only for weighted samples to avoid bias
        value_directional_acc_per_sample = ((value_preds > 0) == (value_targets > 0)).float()
        
        if num_weighted_samples > 0:
            weighted_value_acc = (value_directional_acc_per_sample * sample_weights).sum() / num_weighted_samples
        else:
            weighted_value_acc = torch.tensor(0.0, device=value_preds.device, dtype=value_preds.dtype)
        
        # For logging purposes, calculate unweighted value loss for comparison
        value_loss_unweighted = F.mse_loss(value_preds, value_targets)
        if not torch.isfinite(value_loss_unweighted):
            value_loss_unweighted = torch.tensor(0.0, device=value_preds.device)
        
        value_sample_fraction = sample_weights.mean()
        
        # Log value metrics
        on_step = (phase == 'train')
        self.log(f'{phase}_value_loss_weighted', weighted_value_loss, on_step=on_step, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log(f'{phase}_value_loss_unweighted', value_loss_unweighted, on_step=on_step, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
        self.log(f'{phase}_value_directional_acc', weighted_value_acc, on_step=on_step, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log(f'{phase}_value_sample_fraction', value_sample_fraction, on_step=on_step, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
        
        return weighted_value_loss



    def _calculate_policy_loss(self, policy_preds, policy_targets, batch, phase):
        """
        Calculates policy loss for edge-level predictions with a variable number
        of candidates per graph, correctly handling the ignore_index for targets.
        
        Args:
            policy_preds (Tensor): Flat tensor of logits for all candidate moves in the batch.
                                Shape: [total_moves_in_batch].
            policy_targets (Tensor): Tensor of target indices for each graph. Contains -100 for ignored samples.
                                    Shape: [batch_size].
            batch (torch_geometric.data.Batch): The batch object, containing `move_batch_idx`.
            phase (str): 'train' or 'val'.
        """
        # --- 1. Reshape Predictions from Flat to Dense Batch ---
        # `policy_preds` is flat, e.g., shape [7635]. We need to group logits by graph.
        # `batch.move_batch_idx` maps each of the 7635 moves to its graph index (0-127).
        # `to_dense_batch` creates a padded tensor of shape [batch_size, max_moves_in_batch].
        dense_policy_preds, mask = to_dense_batch(policy_preds, batch.move_batch_idx)
        
        # Set logits for padded (non-existent) moves to a very low value so they don't affect softmax.
        dense_policy_preds[~mask] = -1e9

        # --- 2. Calculate Loss ---
        # `F.cross_entropy` will automatically ignore any target where the label is -100.
        policy_loss = F.cross_entropy(dense_policy_preds, policy_targets)

        if not torch.isfinite(policy_loss):
            # Handle rare cases of non-finite loss if they occur
            return torch.tensor(0.0, device=policy_preds.device, requires_grad=True)

        # --- 3. Calculate Accuracy Correctly (Ignoring -100) ---
        # Create a mask to select only the valid samples (where target is not -100)
        valid_mask = (policy_targets != -100)
        num_valid_samples = valid_mask.sum()

        if num_valid_samples > 0:
            # Get predictions and targets ONLY for the valid samples
            valid_preds = dense_policy_preds[valid_mask]
            valid_targets = policy_targets[valid_mask]

            # Top-1 Accuracy
            pred_indices = valid_preds.argmax(dim=1)
            top_1_acc = (pred_indices == valid_targets).float().mean()

            # Top-k Accuracy
            vocab_size = valid_preds.size(1) # Max moves for these valid samples
            top_5_k = min(5, vocab_size)
            top_10_k = min(10, vocab_size)

            if top_5_k > 1:
                in_top_5 = valid_preds.topk(top_5_k, dim=1).indices.eq(valid_targets.view(-1, 1)).any(dim=1)
                top_5_acc = in_top_5.float().mean()
            else:
                top_5_acc = top_1_acc
            
            if top_10_k > 1:
                in_top_10 = valid_preds.topk(top_10_k, dim=1).indices.eq(valid_targets.view(-1, 1)).any(dim=1)
                top_10_acc = in_top_10.float().mean()
            else:
                top_10_acc = top_1_acc
        else:
            # If the batch has no valid samples, accuracy is 0
            top_1_acc = top_5_acc = top_10_acc = torch.tensor(0.0, device=policy_preds.device)

        # --- 4. Log Metrics ---
        # The conceptual batch size is the number of graphs, not the number of valid samples.
        batch_size = policy_targets.size(0)
        on_step = (phase == 'train')
        
        self.log(f'{phase}_policy_loss', policy_loss, on_step=on_step, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log(f'{phase}_top_1_acc', top_1_acc, on_step=on_step, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log(f'{phase}_top_5_acc', top_5_acc, on_step=on_step, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
        self.log(f'{phase}_top_10_acc', top_10_acc, on_step=on_step, on_epoch=False, logger=True, batch_size=batch_size)
        
        return policy_loss

    def _shared_step(self, batch, phase):
        """Shared logic for training and validation steps to avoid code duplication."""
        # Input validation
        if not hasattr(batch, 'value') or not hasattr(batch, 'policy'):
            raise ValueError("Batch must have 'value' and 'policy' attributes")
        
        outputs = self(batch)
        
        # Validate model outputs
        if not isinstance(outputs, dict) or "value" not in outputs or "policy" not in outputs:
            raise ValueError("Model output must be a dictionary containing 'value' and 'policy' keys")

        # Validate consistent batch sizes and shapes
        batch_size = batch.value.size(0)
        # Check the value head output size, as it should ALWAYS match the batch size.
        if outputs["value"].size(0) != batch_size:
            raise ValueError(f"Value output batch size ({outputs['value'].size(0)}) does not match input batch size ({batch_size})")

        # Calculate weighted value loss
        weighted_value_loss = self._calculate_weighted_value_loss(
            value_preds=outputs["value"],
            value_targets=batch.value,
            batch=batch,
            phase=phase
        )

        # Calculate policy loss
        policy_loss = self._calculate_policy_loss(
            policy_preds=outputs["policy"],
            policy_targets=batch.policy,
            batch=batch,
            phase=phase
        )
        
        # Combine losses
        loss = (self.hparams.policy_weight * policy_loss +
                self.hparams.value_weight * weighted_value_loss)

        # Log combined loss
        self.log(f'{phase}_loss', loss, on_step=(phase == 'train'), on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        return loss

    def training_step(self, batch, batch_idx):
        # GPU-synchronized timing for accurate measurement
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        
        loss = self._shared_step(batch, 'train')

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_time = time.time() - start_time
        
        # Log timing
        self.log('train_step_time', step_time, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch.value.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        `torch.no_grad()` is automatically applied by the Trainer.
        """
        return self._shared_step(batch, 'val')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                              lr=self.hparams.learning_rate,
                              weight_decay=self.hparams.weight_decay)
        return optimizer