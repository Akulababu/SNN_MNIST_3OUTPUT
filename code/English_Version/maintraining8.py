"""
Main Training Module for SNN
This is the main training script that implements the following features:
- Initializes and configures the SNN model
- Sets up training parameters and optimization
- Manages the training loop
- Handles weight quantization and pruning
- Monitors training progress and performance
"""

import torch
import torch.nn as nn
import numpy as np
from indataprocess8 import MNISTDataProcessor
import snntorch as snn
from gennet8 import genet
from Trainloop8 import train_model, create_optimizer_with_layer_lrs
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F  
from gennet8 import QuantLayer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and process dataset
batch_size = 128
data_processor = MNISTDataProcessor(batch_size=batch_size)
processed_train_loader, processed_test_loader = data_processor.get_processed_loaders()

# Initialize model parameters
num_inputs = 49  
num_hidden = [24]  
num_outputs = 3   
beta_values = [0,0]  
thresholds = [1.99, 1.99]  

# Define quantization start epoch
quant_start_epoch = 70 

# Define pruning configuration for each layer using dictionary
pruning_config = {
    0: {
        'input_connections': 8,  # Strictly limit to 8 input connections
        'start_epoch': 30
    },
    1: {  # Configuration for output layer (previously middle layer)
        'input_connections': 8,  # Limit each output neuron to maximum 8 input connections
        'start_epoch': 50
    }
}

# Custom optimizer configuration
optimizer_fn = lambda params: torch.optim.AdamW(
    params, # Network parameters to optimize
    lr=0.05,  # Initial learning rate (previously 0.005)
    weight_decay=0, # L2 regularization coefficient (previously 0.0001)
    betas=(0.9, 0.999) # Momentum coefficients for gradient moving average
)

def combined_loss_with_constraints(spikes, mem_potentials, targets):
    
    """ Combined loss function considering spikes, membrane potentials, and weight constraints"""
    
    spike_loss = nn.CrossEntropyLoss()(spikes, targets)
    norm_mem = F.softmax(mem_potentials, dim=1)
    mem_loss = nn.CrossEntropyLoss()(norm_mem, targets)

    total_loss = 0.6 * spike_loss + 0.4 * mem_loss
    return total_loss   

# Training parameters
num_epochs = 300
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

# Create model
model, forward_fn = genet(
    num_inputs=num_inputs,
    num_hidden=num_hidden,
    num_outputs=num_outputs,
    beta_values=beta_values,
    thresholds=thresholds,
    optimizer_fn=optimizer_fn,
    pruning_config=pruning_config,
    quant_start_epoch=quant_start_epoch
)

# Create optimizer
optimizer = create_optimizer_with_layer_lrs(
    model, 
    optimizer_fn, 
)

# Add scheduler configuration
scheduler_config = {
    'enabled': False,  # Whether to enable scheduler
    'type': 'onecycle',  # Scheduler type
    'params': {
        'max_lr': 0.02,
        'div_factor': 10,
        'pct_start': 0.3,
    }
}

# Create scheduler
if scheduler_config['enabled']:
    if scheduler_config['type'] == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            **scheduler_config['params'],
            epochs=num_epochs,
            steps_per_epoch=len(processed_train_loader),
        )
    else:
        scheduler = None
else:
    scheduler = None

# Train model
best_accuracy = train_model(
    net=model,
    forward_fn=forward_fn,
    train_loader=processed_train_loader,
    test_loader=processed_test_loader,
    num_epochs=num_epochs,
    start_save_epoch=101,
    device=device,
    loss_fn=combined_loss_with_constraints,
    eval_interval=1,
    scheduler_config=scheduler_config,
    optimizer_fn=optimizer_fn,
)