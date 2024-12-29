import numpy as np

def load_matrices_safely(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        matrices = {}
        current_matrix = None
        data_lines = []
        
        for line in lines:
            if "Layer" in line and "Matrix" in line:
                current_matrix = f"Layer {line.split('Layer')[1].split()[0]}"
                data_lines = []
                continue
                
            if current_matrix and "Matrix Statistics:" in line:
                if data_lines:
                    matrices[current_matrix] = np.loadtxt(data_lines)
                continue
                
            if not line.strip() or line.strip()[0] in ['=', 'B']:
                continue
                
            try:
                [float(x) for x in line.strip().split()]
                data_lines.append(line)
            except ValueError:
                continue
        
        if not matrices:
            raise ValueError("No valid matrix data found")
            
        return matrices
        
    except Exception as e:
        print(f"Error reading {filename}:")
        print(f"Specific error: {str(e)}")
        return None

def process_matrices_reverse(weight_matrix_layer0, weight_matrix_layer2):
    with open('reverse_neuron_connections.txt', 'w', encoding='utf-8') as output_file:
        # Layer 1 Analysis (receiving from Layer 0)
        output_file.write("Layer 1 Input Connections Analysis:\n")
        output_file.write("=" * 50 + "\n")
        
        # Process Layer 1 inputs from Layer 0
        for i in range(weight_matrix_layer0.shape[0]):  # 遍历Layer 1的神经元
            connections = weight_matrix_layer0[i, :]
            fanin = np.count_nonzero(connections)
            if fanin > 0:
                positive_connections = np.sum(connections > 0)
                negative_connections = np.sum(connections < 0)
                output_file.write(f"Layer 1 Neuron {i+1} fanin: {fanin} (+{positive_connections}, -{negative_connections})\n")
                for j in range(weight_matrix_layer0.shape[1]):  # 遍历Layer 0的神经元
                    if weight_matrix_layer0[i,j] != 0:
                        weight = weight_matrix_layer0[i,j]
                        sign = "+" if weight > 0 else "-"
                        output_file.write(f"Layer 1 Neuron {i+1} receives from Layer 0 neuron {j+1} ({sign}1)\n")
                output_file.write("\n")
        
        # Layer 3 Analysis (receiving from Layer 2)
        output_file.write("\nLayer 3 Input Connections Analysis:\n")
        output_file.write("=" * 50 + "\n")
        
        # Process Layer 3 inputs from Layer 2
        for i in range(weight_matrix_layer2.shape[0]):  # 遍历Layer 3的神经元
            connections = weight_matrix_layer2[i, :]
            fanin = np.count_nonzero(connections)
            if fanin > 0:
                positive_connections = np.sum(connections > 0)
                negative_connections = np.sum(connections < 0)
                output_file.write(f"Layer 3 Neuron {i+1} fanin: {fanin} (+{positive_connections}, -{negative_connections})\n")
                for j in range(weight_matrix_layer2.shape[1]):  # 遍历Layer 2的神经元
                    if weight_matrix_layer2[i,j] != 0:
                        weight = weight_matrix_layer2[i,j]
                        sign = "+" if weight > 0 else "-"
                        output_file.write(f"Layer 3 Neuron {i+1} receives from Layer 2 neuron {j+1} ({sign}1)\n")
                output_file.write("\n")
        
        # Isolated Neurons Analysis
        output_file.write("\nIsolated Neurons Analysis:\n")
        output_file.write("=" * 50 + "\n")
        
        # Layer 1 isolated neurons (no input connections)
        isolated_layer1 = []
        for i in range(weight_matrix_layer0.shape[0]):
            if np.count_nonzero(weight_matrix_layer0[i, :]) == 0:
                isolated_layer1.append(i+1)
        
        # Layer 2 isolated neurons (no input from layer1 and no output to layer3)
        isolated_layer2 = []
        for j in range(weight_matrix_layer2.shape[1]):
            if np.count_nonzero(weight_matrix_layer2[:, j]) == 0:
                isolated_layer2.append(j+1)
        
        # Layer 3 isolated neurons (no input connections)
        isolated_layer3 = []
        for i in range(weight_matrix_layer2.shape[0]):
            if np.count_nonzero(weight_matrix_layer2[i, :]) == 0:
                isolated_layer3.append(i+1)
        
        # Write isolated neurons results
        output_file.write(f"Layer 1 isolated neurons: {len(isolated_layer1)}\n")
        if isolated_layer1:
            output_file.write(f"Isolated neurons: {isolated_layer1}\n")
        
        output_file.write(f"\nLayer 2 isolated neurons: {len(isolated_layer2)}\n")
        if isolated_layer2:
            output_file.write(f"Isolated neurons: {isolated_layer2}\n")
        
        output_file.write(f"\nLayer 3 isolated neurons: {len(isolated_layer3)}\n")
        if isolated_layer3:
            output_file.write(f"Isolated neurons: {isolated_layer3}\n")
        
        total_isolated = len(isolated_layer1) + len(isolated_layer2) + len(isolated_layer3)
        output_file.write(f"\nTotal isolated neurons: {total_isolated}\n")

    print('Reverse neuron connection information and isolated neurons analysis has been written to reverse_neuron_connections.txt')

try:
    # 加载权重矩阵
    matrices = load_matrices_safely('final_weights.txt')
    
    if matrices is None:
        print("Could not proceed due to data loading errors.")
        exit(1)
    
    # 获取两个层的权重矩阵
    weight_matrix_layer0 = matrices['Layer 0']
    weight_matrix_layer2 = matrices['Layer 2']
    
    # 确保权重被限制在 -1, 0, 或 1
    weight_matrix_layer0 = np.clip(np.round(weight_matrix_layer0), -1, 1)
    weight_matrix_layer2 = np.clip(np.round(weight_matrix_layer2), -1, 1)

    process_matrices_reverse(weight_matrix_layer0, weight_matrix_layer2)

except Exception as e:
    print(f"An error occurred: {str(e)}") 