"""
Neural Network Training Loop Module
This code implements the training process with the following features:
- Manages the main training loop
- Handles batch processing and optimization
- Monitors training metrics and progress
- Implements learning rate scheduling
- Saves best model weights
- Evaluates model performance
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score
from gennet8 import QuantLayer
import torch.nn.functional as F  
from torch.optim.lr_scheduler import OneCycleLR 

class TrainingLogger:
    def __init__(self, start_save_epoch=0):

        """Initialize training logger"""

        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.start_save_epoch = start_save_epoch

    def log_epoch(self, epoch, num_epochs, train_loss, train_accuracy, 
                  test_loss=None, test_accuracy=None, net=None, class_accuracies=None):
        
        """Log training information for each epoch"""
        # Print current epoch training and testing accuracy
        print(f"Epoch [{epoch}/{num_epochs}] - Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")
        
        # Update best accuracy, only save after start_save_epoch
        if test_accuracy is not None and epoch >= self.start_save_epoch:
            if test_accuracy > self.best_accuracy:
                self.best_accuracy = float(test_accuracy)
                self.best_epoch = epoch
                print(f"New Best Accuracy: {self.best_accuracy:.2f}% at epoch {self.best_epoch}")
                # Save best weights to file
                self.save_best_weights(net)

    def save_best_weights(self, net):

        """Save best weights to file """

        with open("training_results_weights.txt", 'w') as f:
            f.write(f"\nBest Model Weights (Accuracy: {self.best_accuracy:.2f}%):\n")
            f.write("=" * 50 + "\n\n")
            
            for i, layer in enumerate(net.layers):
                if isinstance(layer, QuantLayer):
                    # Get quantized weights instead of raw weights
                    weights = layer.quantize_weights()
                    f.write(f"Layer {i} Weights Matrix ({weights.size(0)}x{weights.size(1)}):\n")
                    f.write("=" * 50 + "\n")
                    
                    # Print weight matrix
                    for row in weights:
                        row_str = ' '.join([f"{x:g}" for x in row])
                        f.write(row_str + '\n')
                    
                    f.write('\n')
                    
                    # Calculate statistics
                    total = weights.numel()
                    ones = (weights == 1).sum().item()
                    neg_ones = (weights == -1).sum().item()
                    zeros = (weights == 0).sum().item()
                    
                    f.write("Matrix Statistics:\n")
                    f.write(f"Total elements: {total}\n")
                    f.write(f"Number of 1s: {ones} ({100*ones/total:.2f}%)\n")
                    f.write(f"Number of -1s: {neg_ones} ({100*neg_ones/total:.2f}%)\n")
                    f.write(f"Number of 0s: {zeros} ({100*zeros/total:.2f}%)\n")
                    f.write("\n\n")

    def log_weights(self, net, epoch, first_write=False):
        # Remove all logging operations
        pass

def evaluate_spikes(spikes, targets):
    """Evaluate if network output spikes are correct"""
    batch_size = spikes.size(0)
    correct = 0
    
    for i in range(batch_size):
        spike = spikes[i]
        # Check if only one neuron fires
        if spike.sum() == 1 and spike.max() == 1:
            pred = spike.argmax().item()
            if pred == targets[i].item():
                correct += 1
    
    return correct

def evaluate_spikes_detailed(spikes, targets, num_classes=5):
    """
    Evaluate network output spikes with detailed statistics
    Args:
        spikes: Network output spikes
        targets: Target labels
        num_classes: Number of classes
    Returns:
        tuple: (total_correct, class_accuracies)
            - total_correct: Number of correct predictions
            - class_accuracies: Dictionary with per-class statistics
    """
    batch_size = spikes.size(0)
    class_stats = {i: {'correct': 0, 'total': 0} for i in range(num_classes)}
    total_correct = 0
    
    for i in range(batch_size):
        spike = spikes[i]
        target = targets[i].item()
        
        # Update class totals
        class_stats[target]['total'] += 1
        
        # Check if prediction is valid and correct
        if spike.sum() == 1 and spike.max() == 1:
            pred = spike.argmax().item()
            if pred == target:
                total_correct += 1
                class_stats[target]['correct'] += 1
    
    # Calculate per-class accuracies
    class_accuracies = {}
    for class_idx, stats in class_stats.items():
        if stats['total'] > 0:
            accuracy = 100.0 * stats['correct'] / stats['total']
            class_accuracies[class_idx] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total']
            }
    
    return total_correct, class_accuracies

def create_optimizer_with_layer_lrs(net, optimizer_fn):
    """ Create optimizer with same learning rate for all layers """
    params = []
    
    for layer in net.layers:
        if hasattr(layer, 'parameters'):
            params.extend(layer.parameters())
    
    return optimizer_fn(params)

def train_model(
        net,
        forward_fn,
        train_loader,
        test_loader,
        num_epochs=10,
        device=None,
        optimizer_fn=None,
        loss_fn=None,
        eval_interval=1,
        scheduler_config=None,
        start_save_epoch=0,
):
    
    """Main training loop steps"""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    # Create optimizer with unified learning rate
    optimizer = create_optimizer_with_layer_lrs(net, optimizer_fn)
    
    # Create scheduler based on configuration
    scheduler = None
    if scheduler_config and scheduler_config['enabled']:
        if scheduler_config['type'] == 'onecycle':
            scheduler = OneCycleLR(
                optimizer,
                **scheduler_config['params'],
                epochs=num_epochs,
                steps_per_epoch=len(train_loader)
            )
    
    logger = TrainingLogger(start_save_epoch=start_save_epoch)

    for epoch in range(num_epochs):
        net.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        logger.log_weights(net, epoch + 1, first_write=(epoch == 0))
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device).requires_grad_(True)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            spikes, mem_potentials = forward_fn(data)
            
            # Pass net and constraint_weight to loss function
            loss = loss_fn(spikes, mem_potentials, targets)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update scheduler if enabled and is OneCycleLR
            if scheduler and scheduler_config['type'] == 'onecycle':
                scheduler.step()

            # Calculate training accuracy
            train_total += targets.size(0)
            train_correct += evaluate_spikes(spikes, targets)
            train_loss += loss.item()

            if batch_idx % 10 == 0:
                pass
        
        # Update epoch counter for all QuantLayers at end of epoch
        for layer in net.layers:
            if isinstance(layer, QuantLayer) and hasattr(layer, 'on_epoch_end'):
                layer.on_epoch_end()

        # Update scheduler if enabled and not OneCycleLR
        if scheduler and scheduler_config['type'] != 'onecycle':
            scheduler.step()

        if (epoch + 1) % eval_interval == 0:
            net.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            all_spikes = []
            all_targets = []
            
            with torch.no_grad():
                for data, targets in test_loader:
                    data = data.to(device)
                    targets = targets.to(device)
                    
                    spikes, mem_potentials = forward_fn(data)
                    
                    # Pass net and constraint_weight to loss function
                    loss = loss_fn(spikes, mem_potentials, targets)
                    test_loss += loss.item()
                    
                    test_total += targets.size(0)
                    test_correct += evaluate_spikes(spikes, targets)
                    
                    all_spikes.append(spikes)
                    all_targets.append(targets)
            
            # Calculate metrics
            train_accuracy = 100.0 * train_correct / train_total
            test_accuracy = 100.0 * test_correct / test_total
            train_loss = train_loss / len(train_loader)
            test_loss = test_loss / len(test_loader)
            
            # Calculate per-class accuracies
            all_spikes = torch.cat(all_spikes)
            all_targets = torch.cat(all_targets)
            class_accuracies = evaluate_spikes_detailed(all_spikes, all_targets)
            
            # Log epoch results
            logger.log_epoch(
                epoch + 1,
                num_epochs,
                train_loss,
                train_accuracy,
                test_loss,
                test_accuracy,
                net,
                class_accuracies
            )

    print(f"\nTraining completed. Final Best Accuracy: {logger.best_accuracy:.2f}% at epoch {logger.best_epoch}")
    
    return float(logger.best_accuracy)

class ProcessedDataLoader:
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
