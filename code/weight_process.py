def process_scientific_notation(value):
    """Convert scientific notation to simple -1, 0, 1"""
    try:
        if 'e' in value.lower():
            num = float(value)
            if abs(num) < 1e-10:
                return 0
            elif abs(abs(num) - 1) < 1e-10:
                return 1 if num > 0 else -1
            return num
        return float(value)
    except ValueError:
        print(f"Warning: Could not convert value: {value}")
        return 0

def is_data_line(line):
    """Check if line contains data"""
    return any('e' in word.lower() for word in line.split())

def parse_weights_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as out:
        current_matrix = []
        matrix_name = ""
        processing_matrix = False
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            if "Best Model Weights" in line:
                out.write(line + "\n")
                continue
            
            if "Layer" in line and "Matrix" in line:
                if current_matrix:
                    write_matrix(out, matrix_name, current_matrix)
                matrix_name = line
                current_matrix = []
                processing_matrix = True
                out.write("\n" + "=" * 50 + "\n")
                out.write(f"{matrix_name}\n")
                out.write("=" * 50 + "\n\n")
                continue
            
            if line.startswith("=="):
                continue
            
            if processing_matrix and is_data_line(line):
                try:
                    values = line.split()
                    processed_values = [process_scientific_notation(v) for v in values]
                    current_matrix.append(processed_values)
                except Exception as e:
                    print(f"Warning: Skipping problematic line: {line}")
                    continue
        
        if current_matrix:
            write_matrix(out, matrix_name, current_matrix)

def write_matrix(file, name, matrix):
    """Write matrix to file"""
    for row in matrix:
        line = " ".join([f"{int(val):2d}" for val in row])
        file.write(line + "\n")
    
    total = sum(len(row) for row in matrix)
    ones = sum(sum(1 for val in row if val == 1) for row in matrix)
    minus_ones = sum(sum(1 for val in row if val == -1) for row in matrix)
    zeros = sum(sum(1 for val in row if val == 0) for row in matrix)
    
    file.write(f"\nMatrix Statistics:\n")
    file.write(f"Total elements: {total}\n")
    file.write(f"Number of 1s: {ones} ({ones/total*100:.2f}%)\n")
    file.write(f"Number of -1s: {minus_ones} ({minus_ones/total*100:.2f}%)\n")
    file.write(f"Number of 0s: {zeros} ({zeros/total*100:.2f}%)\n")

if __name__ == "__main__":
    parse_weights_file("training_results_weights.txt", "final_weights.txt")
    print("Conversion complete! Results saved to final_weights.txt")
