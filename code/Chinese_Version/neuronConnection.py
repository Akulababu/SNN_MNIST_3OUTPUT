import numpy as np

def load_data_safely(filename):
    try:
        # Skip any header rows by finding the first row that contains only numbers
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        data_lines = []
        for line in lines:
            # Skip empty lines or lines starting with special characters
            if not line.strip() or line.strip()[0] in ['=', 'B']:
                continue
            # Try to convert all items to float to verify it's a data line
            try:
                [float(x) for x in line.strip().split()]
                data_lines.append(line)
            except ValueError:
                continue
                
        # Convert the clean data to numpy array
        return np.loadtxt(data_lines)
    except Exception as e:
        print(f"Error reading {filename}:")
        print(f"Specific error: {str(e)}")
        print("\nPlease ensure the file contains properly formatted numeric data.")
        return None

def load_matrices_safely(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        matrices = {}
        current_matrix = None
        data_lines = []
        
        for line in lines:
            # 检测新矩阵的开始
            if "Layer" in line and "Matrix" in line:
                current_matrix = f"Layer {line.split('Layer')[1].split()[0]}"
                data_lines = []
                continue
                
            # 如果遇到统计信息，保存当前矩阵
            if current_matrix and "Matrix Statistics:" in line:
                if data_lines:
                    matrices[current_matrix] = np.loadtxt(data_lines)
                continue
                
            # 跳过头部和空行
            if not line.strip() or line.strip()[0] in ['=', 'B']:
                continue
                
            # 尝试转换为数值数据
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

def process_matrices(weight_matrix_layer0, weight_matrix_layer2):
    with open('neuron_connections.txt', 'w', encoding='utf-8') as output_file:
        # Layer 0 Analysis
        output_file.write("Layer 0 Weights Matrix Analysis:\n")
        output_file.write("=" * 50 + "\n")
        
        # Process Layer 0
        for j in range(weight_matrix_layer0.shape[1]):  # 遍历目标神经元
            fanout = np.count_nonzero(weight_matrix_layer0[:, j])
            if fanout > 0:
                output_file.write(f"Neuron {j+1} fanout: {fanout}\n")
                for i in range(weight_matrix_layer0.shape[0]):  # 遍历源神经元
                    if weight_matrix_layer0[i,j] != 0:
                        weight = weight_matrix_layer0[i,j]
                        sign = "+" if weight > 0 else "-"
                        output_file.write(f"Neuron {j+1} connects to next layer neuron {i+1} ({sign}1)\n")
                output_file.write("\n")
        
        # Layer 2 Analysis
        output_file.write("\nLayer 2 Weights Matrix Analysis:\n")
        output_file.write("=" * 50 + "\n")
        
        # Process Layer 2
        for j in range(weight_matrix_layer2.shape[1]):
            fanout = np.count_nonzero(weight_matrix_layer2[:, j])
            if fanout > 0:
                output_file.write(f"Neuron {j+1} fanout: {fanout}\n")
                for i in range(weight_matrix_layer2.shape[0]):
                    if weight_matrix_layer2[i,j] != 0:
                        weight = weight_matrix_layer2[i,j]
                        sign = "+" if weight > 0 else "-"
                        output_file.write(f"Neuron {j+1} connects to next layer neuron {i+1} ({sign}1)\n")
                output_file.write("\n")
        
        # Dead Neurons Analysis
        output_file.write("\nDead Neurons Analysis:\n")
        output_file.write("=" * 50 + "\n")
        
        # Layer 0 dead neurons
        dead_neurons_layer0 = []
        for j in range(weight_matrix_layer0.shape[1]):
            if np.count_nonzero(weight_matrix_layer0[:, j]) == 0:
                dead_neurons_layer0.append(j+1)
        
        # Layer 1 dead neurons
        dead_neurons_layer1 = []
        for j in range(weight_matrix_layer2.shape[1]):
            if np.count_nonzero(weight_matrix_layer2[:, j]) == 0:
                dead_neurons_layer1.append(j+1)
        
        # Layer 2 dead neurons
        dead_neurons_layer2 = []
        for i in range(weight_matrix_layer2.shape[0]):
            if np.count_nonzero(weight_matrix_layer2[i]) == 0:
                dead_neurons_layer2.append(i+1)
        
        # Write dead neurons results
        output_file.write(f"Layer 0 dead neurons: {len(dead_neurons_layer0)}\n")
        if dead_neurons_layer0:
            output_file.write(f"Dead neurons: {dead_neurons_layer0}\n")
        
        output_file.write(f"\nLayer 1 dead neurons: {len(dead_neurons_layer1)}\n")
        if dead_neurons_layer1:
            output_file.write(f"Dead neurons: {dead_neurons_layer1}\n")
        
        output_file.write(f"\nLayer 2 dead neurons: {len(dead_neurons_layer2)}\n")
        if dead_neurons_layer2:
            output_file.write(f"Dead neurons: {dead_neurons_layer2}\n")
        
        total_dead = len(dead_neurons_layer0) + len(dead_neurons_layer1) + len(dead_neurons_layer2)
        output_file.write(f"\nTotal dead neurons: {total_dead}\n")

    print('Neuron connection information and dead neurons analysis has been written to neuron_connections.txt')

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

    process_matrices(weight_matrix_layer0, weight_matrix_layer2)

except Exception as e:
    print(f"An error occurred: {str(e)}")