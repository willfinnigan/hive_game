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
                 min_step_for_value=6,
                 task_weights={'policy': 1.0, 'value': 0.2}):
        
        super().__init__()
        # Save hyperparameters but ignore the model to avoid the warning
        self.save_hyperparameters(ignore=['model'])
        self.model = model

        self.loss_functions = {
            'value': self.value_loss,
            'policy': self.policy_loss,
            'mobile_pieces': self.mobile_pieces_loss}

    def forward(self, batch):
        return self.model(batch)

    def value_loss(self, predictions, targets, batch, phase):
        """Calculate value loss, ignoring certain samples, using Huber Loss."""
        batch_size = predictions.size(0)

        # Create mask: only train on steps >= min_step and non-draw games
        step_mask = batch.step >= self.hparams.min_step_for_value
        value_mask = batch.value != 0
        mask = step_mask & value_mask

        # Ensure we don't try to calculate loss on an empty set of samples
        if mask.sum() == 0:
            loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
            acc = torch.tensor(0.0, device=predictions.device)
        else:
            filtered_preds = predictions[mask]
            filtered_targets = targets[mask]

            loss = F.mse_loss(filtered_preds, filtered_targets)
            acc = ((filtered_preds > 0) == (filtered_targets > 0)).float().mean()

        # Log metrics
        on_step = (phase == 'train')
        num_valid_samples = mask.sum().item()
        if num_valid_samples == 0:
            num_valid_samples = 1 # Avoid division by zero if batch is empty
        
        # Log metrics
        on_step = (phase == 'train')
        self.log(f'{phase}_value_loss', loss,
                on_step=on_step, on_epoch=True, batch_size=batch_size)
        self.log(f'{phase}_value_directional_acc', acc,
                on_step=on_step, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f'{phase}_valid_sample_count', mask.sum().float(), on_step=True)
        
        return loss

    def mobile_pieces_loss(self, predictions, targets, batch, phase):
        """Calculate mobile pieces loss"""
        loss = F.mse_loss(predictions, targets)
        
        self.log(f'{phase}_mobile_pieces_loss', loss, 
                on_step=(phase == 'train'), on_epoch=True, batch_size=batch.value.size(0))
        
        return loss

    def policy_loss(self, predictions, targets, batch, phase):
        """Calculate policy loss for variable number of moves per graph"""
        # Convert flat predictions to dense batch format
        dense_preds, mask = to_dense_batch(predictions, batch.move_batch_idx)
        dense_preds[~mask] = -1e9  # Mask padded positions
        
        # Calculate loss
        loss = F.cross_entropy(dense_preds, targets)
        
        # Calculate accuracy
        top_1_acc = (dense_preds.argmax(dim=1) == targets).float().mean()
        
        # Top-k accuracy
        vocab_size = dense_preds.size(1)
        top_5_acc = top_1_acc  # Default to top-1
        top_10_acc = top_1_acc
        
        if vocab_size >= 5:
            top_5_hits = dense_preds.topk(5, dim=1).indices.eq(targets.view(-1, 1)).any(dim=1)
            top_5_acc = top_5_hits.float().mean()
        
        if vocab_size >= 10:
            top_10_hits = dense_preds.topk(10, dim=1).indices.eq(targets.view(-1, 1)).any(dim=1)
            top_10_acc = top_10_hits.float().mean()
        
        # Log metrics
        on_step = (phase == 'train')
        self.log(f'{phase}_policy_loss', loss, on_step=on_step, on_epoch=True, prog_bar=True, batch_size=targets.size(0))
        self.log(f'{phase}_top_1_acc', top_1_acc, on_step=on_step, on_epoch=True, prog_bar=True, batch_size=targets.size(0))
        self.log(f'{phase}_top_5_acc', top_5_acc, on_step=on_step, on_epoch=True, batch_size=targets.size(0))
        self.log(f'{phase}_top_10_acc', top_10_acc, on_step=on_step, on_epoch=True, batch_size=targets.size(0))
        
        return loss

    def step(self, batch, phase):
        """Shared logic for training and validation steps to avoid code duplication."""


        outputs = self(batch)

        # Iterate through all tasks defined in the configuration
        total_loss = torch.tensor(0.0, device=self.device)
        for task_name, weight in self.hparams.task_weights.items():
            if weight == 0:
                continue
            if task_name not in outputs or outputs[task_name] is None:
                continue

            loss_function = self.loss_functions.get(task_name)
            predictions = outputs[task_name]
            targets = getattr(batch, task_name)

            individual_loss = loss_function(predictions=predictions,
                                               targets=targets,
                                               batch=batch,
                                               phase=phase)

            total_loss += weight * individual_loss


        # Log combined loss
        self.log(f'{phase}_loss', total_loss, on_step=(phase == 'train'), on_epoch=True, prog_bar=True, logger=True, batch_size=batch.value.size(0))

        return total_loss

    def training_step(self, batch, batch_idx):
        # GPU-synchronized timing for accurate measurement
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        
        loss = self.step(batch, 'train')

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
        return self.step(batch, 'val')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                              lr=self.hparams.learning_rate,
                              weight_decay=self.hparams.weight_decay)
        return optimizer