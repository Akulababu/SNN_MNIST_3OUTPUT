"""
This script is used to analyze the weight matrices of the final network for the hardware implementation, 
you can get the connection distribution of each neuron, the number of dead neurons, the maximum and 
total fan-out values, and the number of positive and negative connections.
"""
import numpy as np
import re

def read_weight_matrices(file_path):
    """Read three weight matrices from file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Use regular expression to find each matrix section
    matrix_sections = re.findall(r'Layer \d Weights Matrix.*?Matrix Statistics:', 
                               content, re.DOTALL)
    
    matrices = []
    for section in matrix_sections:
        # Extract matrix rows
        matrix_lines = []
        for line in section.split('\n'):
            # Skip title and statistics lines
            if ('===' in line or 'Layer' in line or 'Matrix' in line or 
                not line.strip() or 'Statistics' in line):
                continue
            # Convert each line to a list of numbers
            row = [int(x) for x in line.strip().split() if x in ['0', '1', '-1']]
            if row:  # If the row is not empty
                matrix_lines.append(row)
        
        if matrix_lines:
            matrices.append(np.array(matrix_lines))
    
    # If there are only two matrices, return an empty matrix for the third one
    if len(matrices) == 2:
        matrices.append(np.array([]))
        
    return matrices[0], matrices[1], matrices[2]

def analyze_layer_weights(weight_matrix, layer_name):
    """Analyze weights matrix for a single layer"""
    # Skip analysis if matrix is empty
    if weight_matrix.size == 0:
        print(f"\n=== {layer_name} Analysis Results ===")
        print("No matrix data available for this layer.")
        return
        
    print(f"\n=== {layer_name} Analysis Results ===")
    print(f"Matrix shape: {weight_matrix.shape}")
    
    # Calculate positive and negative counts for each neuron
    positive_counts = np.sum(weight_matrix > 0, axis=1)
    negative_counts = np.sum(weight_matrix < 0, axis=1)
    
    # Calculate fan-out characteristics
    weight_tr = weight_matrix.T
    weight_tr_abs = np.abs(weight_tr)
    fanout = np.sum(weight_tr_abs, axis=1)
    
    # Find dead neurons
    fanout_zeros = np.all(fanout == 0, axis=0)
    fanout_zeros_indexes = np.where(fanout_zeros)[0]
    fanout_zeros_total = np.sum(fanout_zeros)
    
    # Calculate maximum and total fan-out values
    fanout_max = np.max(fanout)
    fanout_total = np.sum(fanout)
    
    # Print results
    print(f"\n1. Fan-out Analysis:")
    print(f"- Maximum fan-out value: {fanout_max}")
    print(f"- Total fan-out value: {fanout_total}")
    print(f"- Number of dead neurons: {fanout_zeros_total}")
    if len(fanout_zeros_indexes) > 0:
        print(f"- Dead neuron indices: {fanout_zeros_indexes}")
    
    print(f"\n2. Connection Analysis:")
    print(f"- Total connections: {weight_matrix.size}")
    print(f"- Positive connections: {np.sum(weight_matrix > 0)} ({100*np.sum(weight_matrix > 0)/weight_matrix.size:.2f}%)")
    print(f"- Negative connections: {np.sum(weight_matrix < 0)} ({100*np.sum(weight_matrix < 0)/weight_matrix.size:.2f}%)")
    print(f"- Zero connections: {np.sum(weight_matrix == 0)} ({100*np.sum(weight_matrix == 0)/weight_matrix.size:.2f}%)")
    
    print("\n3. Connection distribution for each output neuron:")
    for i in range(len(weight_matrix)):
        print(f"Neuron {i}: Positive = {positive_counts[i]}, Negative = {negative_counts[i]}, "
              f"Total connections = {positive_counts[i] + negative_counts[i]}")

def main():
    # Read weight matrices
    layer0_weights, layer2_weights, layer4_weights = read_weight_matrices('final_weights.txt')
    
    # Analyze each layer
    analyze_layer_weights(layer0_weights, "Layer 0 (Input Layer)")
    analyze_layer_weights(layer2_weights, "Layer 2 (Hidden Layer)")
    analyze_layer_weights(layer4_weights, "Layer 4 (Output Layer)")

if __name__ == "__main__":
    main()