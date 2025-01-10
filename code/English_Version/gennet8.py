"""
Neural Network Framework for SNN
This code implements a neural network framework with the following features:
- Ternary weight quantization
- Weight pruning capabilities
- Weight constraints and normalization
- Specialized for SNN architectures
- Supports both training and inference modes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

class TernaryQuantFunction(torch.autograd.Function):
    """Ternary quantization function using Straight-Through Estimator for backpropagation"""
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        output = torch.zeros_like(input)
        output[input > threshold] = 1.0    # Values above threshold become 1
        output[input < -threshold] = -1.0  # Values below negative threshold become -1
        return output

    @staticmethod
    # Backward propagation function using C++ SpikeTorch library, difficult to debug
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None   

class QuantLayer(nn.Module):
    def __init__(self, in_features, out_features, prune_config=None, quant_start_epoch=10, layer_index=0):
        super().__init__()
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Standard linear layer
        self.linear = nn.Linear(in_features, out_features, bias=False).to(self.device)
        
        # Initialize mask
        self.mask = torch.ones(out_features, in_features).to(self.device)
        
        # Quantization control parameters
        self.enable_quantization = False
        self.quant_start_epoch = quant_start_epoch
        
        self.epoch_counter = 0
        self.pruning_counter = 0
        self.prune_config = prune_config
        
        # Dynamic threshold parameters
        self.threshold = None  # Will be dynamically calculated during quantization
        
        # Save layer size information
        self.in_features = in_features
        self.out_features = out_features
        
        # Ternary quantization function
        self.ternary_quant = TernaryQuantFunction.apply
        
        # Add a variable to save fixed thresholds
        self.fixed_threshold = None
        
        # Set layer index
        self.layer_index = layer_index
        
        # Add attributes related to weight constraints
        self.max_positive_weights = 6  # Each output neuron can have up to 6 positive weights
        self.max_negative_weights = 2  # Each output neuron can have up to 2 negative weights
        
        # Use Kaiming initialization instead of uniform initialization
        nn.init.kaiming_normal_(self.linear.weight, 
                              mode='fan_in',  # Use fan_in mode
                              nonlinearity='leaky_relu',
                              a=0.1)  # Smaller slope parameter
        
        # Add weight range restriction
        with torch.no_grad():
            self.linear.weight.data.clamp_(-1, 1)
        
        # Add attributes related to normalization
        self.weight_mean = None
        self.weight_std = None
        self.is_normalized = False
        
    def calculate_threshold(self, weights):
        """Calculate threshold based on layer position"""
        if self.layer_index == 0:  # First layer
            abs_weights = weights.abs()
            
            # Method 4: Adaptive threshold based on weight distribution
            sorted_weights, _ = torch.sort(abs_weights.view(-1))
            target_idx = int(len(sorted_weights) * 0.93)
            window_size = len(sorted_weights) // 20
            start_idx = max(0, target_idx - window_size // 2)
            end_idx = min(len(sorted_weights), target_idx + window_size // 2)
            threshold4 = sorted_weights[start_idx:end_idx].mean()
            
            return threshold4
            
        else:  # Last layer
            # Use smaller threshold to maintain more connections
            mean = weights.abs().mean()
            std = weights.abs().std()
            return 0.1 * mean + 0.05 * std
        
    def to(self, device):
        """Ensure all components are moved to the correct device"""
        super().to(device)
        self.device = device
        self.linear = self.linear.to(device)
        self.mask = self.mask.to(device)
        return self
    
    def enforce_constraints(self):
        with torch.no_grad():
            weights = self.linear.weight.data

            for i in range(self.out_features):  # Iterate over each row
                row = weights[i, :]
                mask_row = self.mask[i, :]  # Get the pruning mask for that row

                # Only consider weights that are not pruned
                valid_weights = row * mask_row

                # Process positive weights
                positive_mask = (valid_weights > 0)
                positive_weights = valid_weights[positive_mask]
                if len(positive_weights) > self.max_positive_weights:
                    # Keep the top max_positive_weights positive weights
                    values, indices = torch.topk(positive_weights, self.max_positive_weights)

                    # Create a new weight vector initialized to 0
                    new_row = torch.zeros_like(row)

                    # Get the indices of the original positive weights and only select those that are not pruned
                    positive_indices = torch.nonzero(positive_mask & (mask_row == 1)).squeeze()
                    selected_indices = positive_indices[indices]
                    new_row[selected_indices] = values

                    # Keep negative weights unchanged but only retain those that are not pruned
                    negative_mask = (valid_weights < 0)
                    new_row[negative_mask] = valid_weights[negative_mask]

                    # Update the row
                    weights[i, :] = new_row

                # Process negative weights
                negative_mask = (valid_weights < 0)
                negative_weights = valid_weights[negative_mask]
                if len(negative_weights) > self.max_negative_weights:
                    # Keep the absolute value of the largest max_negative_weights negative weights
                    values, indices = torch.topk(negative_weights.abs(), self.max_negative_weights)
                    negative_indices = torch.nonzero(negative_mask & (mask_row == 1)).squeeze()
                    selected_indices = negative_indices[indices]

                    # Create a new weight vector keeping positive weights unchanged
                    temp_row = weights[i, :].clone()
                    temp_row[negative_mask] = 0  # Clear all negative weights
                    temp_row[selected_indices] = negative_weights[indices]  # Only retain selected negative weights

                    # Ensure only unpruned weights are updated
                    weights[i, :] = temp_row * mask_row

            # Ensure the weight matrix is updated to meet constraints
            self.linear.weight.data = torch.clamp(weights, -1.0, 1.0)

    def normalize_weights(self):
        """Normalize weights and restrict them to [-1,1]"""
        with torch.no_grad():
            weights = self.linear.weight.data
            self.weight_mean = weights.mean()
            self.weight_std = weights.std()
            normalized_weights = (weights - self.weight_mean) / (self.weight_std + 1e-8)
            normalized_weights = torch.clamp(normalized_weights, -1.0, 1.0)
            self.linear.weight.data = normalized_weights
            self.is_normalized = True       
            
    def forward(self, x):
        # Always normalize
        if not self.is_normalized:
            self.normalize_weights()
        
        # First check if pruning is needed
        if (self.prune_config is not None and 
            self.epoch_counter >= self.prune_config['start_epoch']):
            self.update_mask()
        
        # Always use quantized weights for forward propagation
        self.enforce_constraints()
        weights = self.linear.weight * self.mask
        if self.enable_quantization:
            # Ensure weights are normalized before quantization
            if not self.is_normalized:
                self.normalize_weights()
            self.threshold = self.calculate_threshold(weights)
            w_q = self.ternary_quant(weights, self.threshold)
        else:
            # Even in non-quantized mode, ensure weights are within [-1,1]
            w_q = torch.clamp(weights, -1.0, 1.0)
        
        return F.linear(x, w_q)

    def update_mask(self):
        """Update pruning mask"""
        if self.prune_config is None:
            return
        
        with torch.no_grad():
            weights = self.linear.weight.data.abs()  # Use absolute values
            new_mask = torch.zeros_like(self.mask, device=self.device)
            
            # Prune each output neuron
            for i in range(self.out_features):
                # Get the absolute values of all input weights for that neuron
                w = weights[i]
                # Keep the top k weights
                k = self.prune_config['input_connections']
                # Find the indices of the top k weights
                _, indices = torch.topk(w, k)
                # Set those positions in the mask to 1
                new_mask[i][indices] = 1
            
            # Update mask and apply
            self.mask = new_mask
            self.linear.weight.data *= self.mask

    def on_epoch_end(self):
        """Called at the end of each epoch"""
        self.epoch_counter += 1
        
        if self.epoch_counter == self.quant_start_epoch - 1:
            self.normalize_weights()
        
        if (self.prune_config is not None and 
            self.epoch_counter == self.prune_config['start_epoch']):
            self.update_mask()
            w = self.linear.weight.data * self.mask
        
        if not self.enable_quantization and self.epoch_counter >= self.quant_start_epoch:
            self.enable_quantization = True
            w = self.linear.weight.data * self.mask
            self.threshold = self.calculate_threshold(w)
            w_q = self.ternary_quant(w, self.threshold)
            self.linear.weight.data = w_q

    def quantize_weights(self):
        """Return current weight state (pruned and/or quantized)"""
        with torch.no_grad():
            if self.enable_quantization:
                # Ensure weights are normalized
                if not self.is_normalized:
                    self.normalize_weights()
                self.enforce_constraints()
                # Perform quantization
                self.threshold = self.calculate_threshold(self.linear.weight.data)
                pruned_weights = self.linear.weight * self.mask
                # Quantize weights
                quantized_weights = self.ternary_quant(pruned_weights, self.threshold)
                # Apply quantized weights to model
                self.linear.weight.data.copy_(quantized_weights)
                return self.linear.weight.data
            else:
                # If not quantized, return pruned weights that meet constraints
                self.enforce_constraints()
                return self.linear.weight * self.mask

def genet(num_inputs, num_hidden, num_outputs, beta_values, thresholds, 
          optimizer_fn, pruning_config=None, quant_start_epoch=10):
    
    '''Create and configure SNN network'''
    spike_grad = surrogate.fast_sigmoid(slope=20)
    
    # If no pruning config provided, create default config (no pruning)
    if pruning_config is None:
        pruning_config = {i: None for i in range(len(num_hidden) + 1)}
    
    # Create network
    net = QuantSNNNetwork(
        num_inputs=num_inputs,
        num_hidden=num_hidden,
        num_outputs=num_outputs,
        beta=beta_values,
        threshold=thresholds,
        pruning_config=pruning_config,
        quant_start_epoch=quant_start_epoch
    )
    
    # Single step forward propagation
    def forward_single_step(x):
        net.reset_hidden()
        output = net(x)
        return output

    return net, forward_single_step

class QuantSNNNetwork(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta, threshold, 
                 pruning_config, quant_start_epoch):
        super().__init__()
        
        self.layers = nn.ModuleList()
        delay_between_layers = 30
        
        # Input layer -> First hidden layer
        self.layers.append(QuantLayer(num_inputs, num_hidden[0], 
                                    prune_config=pruning_config.get(0),
                                    quant_start_epoch=quant_start_epoch + delay_between_layers,
                                    layer_index=0))
        
        self.layers.append(snn.Leaky(beta=beta[0],
                                    threshold=threshold[0],
                                    spike_grad=surrogate.fast_sigmoid(slope=20),
                                    reset_mechanism='zero',
                                    init_hidden=True,
                                    output='both'))

        # First hidden layer -> Output layer
        self.layers.append(QuantLayer(num_hidden[0], num_outputs,
                                    prune_config=pruning_config.get(1),
                                    quant_start_epoch=quant_start_epoch,
                                    layer_index=1))
        
        self.layers.append(snn.Leaky(beta=beta[1],
                                    threshold=threshold[1],
                                    spike_grad=surrogate.fast_sigmoid(slope=20),
                                    reset_mechanism='zero',
                                    init_hidden=True,
                                    output='both'))
    
    def forward(self, x):
        # Layer 1 - First QuantLayer
        x1 = self.layers[0](x)
        
        # Layer 2 - First Leaky
        spk1, mem1 = self.layers[1](x1)
        
        # Layer 3 - Second QuantLayer
        x2 = self.layers[2](spk1)
        
        # Layer 4 - Final Leaky
        spk2, mem2 = self.layers[3](x2)
        
        return spk2, mem2
            
    def reset_hidden(self):  
        """Reset hidden states of all layers"""  
        for layer in self.layers:  
            if isinstance(layer, snn.Leaky):  
                # If mem already exists, use its shape to create a new zero tensor  
                if hasattr(layer, 'mem') and layer.mem is not None:  
                    layer.mem = torch.zeros_like(layer.mem)  
                if hasattr(layer, 'spk') and layer.spk is not None:  
                    layer.spk = torch.zeros_like(layer.spk)  