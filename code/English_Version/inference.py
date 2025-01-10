"""
Inference Module for SNN
This code implements the inference process with the following features:
- Loads trained weights from file
- Applies weights to the SNN model
- Performs inference on test data
- Records and analyzes classification results
- Compares results with verification module
"""

import torch
import numpy as np
import snntorch as snn
from gennet8 import genet
from gennet8 import genet, QuantLayer
from indataprocess8 import MNISTDataProcessor

def load_weights_from_file(file_path, device):
   
    """ Load weights from file """
    
    weights = []
    current_matrix = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and non-weight lines
            if not line or line.startswith('=') or line.startswith('Best') or \
               line.startswith('Matrix') or line.startswith('Layer') or \
               line.startswith('Total') or line.startswith('Number'):
                # Save current matrix when encountering new matrix marker
                if current_matrix and (line.startswith('==') or line.startswith('Matrix')):
                    weights.append(torch.tensor(current_matrix, device=device))
                    current_matrix = []
                continue
                
            # Ensure line contains numbers only
            try:
                # Split line and convert to float list
                row_values = [float(x) for x in line.split()]
                if row_values:  # Ensure line is not empty
                    current_matrix.append(row_values)
            except ValueError:
                continue  # Skip lines that can't be converted to numbers
    
    # Add the last matrix if exists
    if current_matrix:
        weights.append(torch.tensor(current_matrix, device=device))
    
    return weights

def apply_weights_to_model(model, weights):
   
    """Apply weights to model layers"""
    
    layer_idx = 0
    for layer in model.layers:
        if hasattr(layer, 'linear'):
            layer.linear.weight.data = weights[layer_idx]
            layer_idx += 1

def run_inference(model, data_loader, device):
    
    """ Run inference and record classification results """
    
    model.eval()
    correct_cases = []
    incorrect_cases = []
    invalid_cases = []  
    total = 0
    
    with torch.no_grad():
        for data, targets in data_loader:
            for i in range(len(data)):
                single_input = data[i:i+1].to(device)
                target = targets[i].item()
                
                model.reset_hidden()
                
                output = None
                for layer_idx, layer in enumerate(model.layers):
                    if isinstance(layer, QuantLayer):
                        single_input = layer(single_input)
                    elif isinstance(layer, snn.Leaky):
                        layer.mem = torch.zeros_like(layer.mem)
                        assert layer.reset_mechanism == 'zero'
                        single_input, mem = layer(single_input)
                
                output = single_input  
                
                output_spikes = output.cpu().numpy().flatten()
                input_pattern = data[i].cpu().numpy().flatten()
                
                if sum(output_spikes) == 1:  
                    pred_digit = output_spikes.argmax() + 2
                    if pred_digit == target + 2:
                        correct_cases.append({
                            'input': input_pattern,
                            'output': output_spikes,
                            'target': target + 2,
                            'type': 'correct'
                        })
                    else:
                        incorrect_cases.append({
                            'input': input_pattern,
                            'output': output_spikes,
                            'target': target + 2,
                            'type': 'incorrect'
                        })
                else:  
                    invalid_cases.append({
                        'input': input_pattern,
                        'output': output_spikes,
                        'target': target + 2,
                        'type': 'invalid'
                    })
                total += 1
    
    return correct_cases, incorrect_cases, invalid_cases

def save_classification_results(correct_cases, incorrect_cases, invalid_cases, total_samples):
   
    """Save classification results to file"""
    
    with open('correct_cases.txt', 'w') as f:
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Correct Classifications: {len(correct_cases)}\n")
        f.write(f"Incorrect Classifications: {len(incorrect_cases)}\n")
        f.write(f"Invalid Classifications: {len(invalid_cases)}\n")
        f.write(f"Accuracy: {100.0 * len(correct_cases) / total_samples:.2f}%\n")
        f.write("=" * 50 + "\n\n")
        
        for case_type, cases in [
            ("CORRECT CLASSIFICATIONS", correct_cases),
            ("INCORRECT CLASSIFICATIONS", incorrect_cases),
            ("INVALID CLASSIFICATIONS", invalid_cases)
        ]:
            f.write(f"=== {case_type} (Total: {len(cases)}) ===\n")
            f.write("=" * 50 + "\n\n")
            
            for idx, case in enumerate(cases, 1):
                f.write(f"=== {case['type'].capitalize()} Classification Case {idx}/{len(cases)} ===\n")
                f.write("Input Pattern (7x7):\n")
                input_matrix = case['input'].reshape(7, 7)
                for row in input_matrix:
                    f.write(" ".join("1" if x > 0.5 else "0" for x in row) + "\n")
                
                f.write(f"Expected Digit: {case['target']}\n")
                if case['type'] != 'invalid':
                    f.write(f"Recognized Digit: {case['output'].argmax() + 2}\n")
                else:
                    f.write("Recognized Digit: Invalid (multiple or no spikes)\n")
                f.write(f"Output Spikes: {case['output'].tolist()}\n")
                f.write("="*50 + "\n")


def main():
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model parameters (same as training)
    num_inputs = 49
    num_hidden = [24]
    num_outputs = 3
    beta_values = [0,0]
    thresholds = [1.99, 1.99]
    
    # Create model
    model, _ = genet(num_inputs, num_hidden, num_outputs, beta_values, thresholds, 
                    optimizer_fn=None, quant_start_epoch=50)
    model = model.to(device)
    
    # Load weights
    weights = load_weights_from_file('final_weights.txt', device)
    apply_weights_to_model(model, weights)
    
    # Prepare data
    data_processor = MNISTDataProcessor(batch_size=64)
    _, test_loader = data_processor.get_processed_loaders()
    
    # Run inference
    correct_cases, incorrect_cases, invalid_cases = run_inference(model, test_loader, device)
    
    # Calculate total samples
    total_samples = sum(len(data) for data, _ in test_loader)
    
    # Save classification results
    save_classification_results(correct_cases, incorrect_cases, invalid_cases, total_samples)
    
    # Print only total cases and accuracy
    print(f"Total cases: {total_samples}")
    print(f"Accuracy: {100.0 * len(correct_cases) / total_samples:.2f}%")

if __name__ == "__main__":
    main()