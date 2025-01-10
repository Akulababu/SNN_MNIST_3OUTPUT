"""
Neural Network Verification Module
This code implements verification functionality with the following features:
- Loads trained model weights from file
- Verifies network predictions against test cases
- Analyzes classification accuracy and errors
- Generates detailed verification reports
- Identifies discrepancies between expected and actual outputs
- Provides statistical analysis of network performance
- Saves verification.txt and verification_differences.txt results to files
"""

import numpy as np

def load_weights(weights_file):
    """Load two weight matrices from the weights file"""
    weights = []
    current_matrix = []
    reading_matrix = False

    with open(weights_file, 'r') as f:
        for line in f:
            line = line.strip()
            if "Layer" in line and "Weights Matrix" in line:
                if current_matrix:
                    weights.append(np.array(current_matrix, dtype=float))
                    current_matrix = []
                reading_matrix = True
                continue

            if reading_matrix:
                if line and all(c in '0123456789-. ' for c in line):
                    row = [float(x) for x in line.split()]
                    current_matrix.append(row)
                elif "Matrix Statistics" in line:
                    weights.append(np.array(current_matrix, dtype=float))
                    current_matrix = []
                    reading_matrix = False

    # Add the last matrix if it exists
    if current_matrix:
        weights.append(np.array(current_matrix, dtype=float))

    return weights[0], weights[1]  # Return the two weight matrices

def parse_input_pattern(pattern_text):
    
    """Convert 7x7 text pattern into a 1x49 input vector"""
    
    pattern = []
    for line in pattern_text.strip().split('\n'):
        pattern.extend([float(x) for x in line.strip().split()])
    return np.array(pattern, dtype=float)

def simple_matrix_nn(input_pattern, weights1, weights2, thresholds):
    
    """Simple matrix-based neural network that mimics SNN behavior"""
    
    input_pattern = input_pattern.astype(float)
    
    # First layer
    hidden_activation = np.dot(weights1, input_pattern)
    hidden_spikes = np.zeros_like(hidden_activation)  # Initialize with zeros
    hidden_spikes[hidden_activation >= thresholds[0]] = 1.0  # Only neurons above threshold fire
    
    # Second layer
    output_activation = np.dot(weights2, hidden_spikes)
    output_spikes = np.zeros_like(output_activation)  # Initialize with zeros
    output_spikes[output_activation >= thresholds[1]] = 1.0  # Only neurons above threshold fire
    
    # Mimic inference.py logic
    if np.sum(output_spikes) != 1:  # If not exactly one output neuron fires
        return np.zeros_like(output_spikes)  # Return all zeros for invalid prediction
        
    return output_spikes

def verify_all_cases(cases_file, weights_file):
   
    """Verify all classification cases"""
    
    weights1, weights2 = load_weights(weights_file)
    total_cases = 0
    correct_predictions = 0
    
    # Counters for different types of cases
    correct_but_failed = 0  # Should be correct but predicted wrong
    incorrect_but_succeeded = 0  # Marked as incorrect but predicted right
    invalid_but_succeeded = 0  # Marked as invalid but predicted right
    
    # Collect all cases and results
    cases_data = {
        'correct_failed': [],
        'incorrect_succeeded': [],
        'invalid_succeeded': []
    }
    
    with open(cases_file, 'r') as f:
        content = f.read()
        
    import re
    case_pattern = re.compile(
        r'=== (Correct|Incorrect|Invalid) Classification.*?Input Pattern \(7x7\):\n((?:\d+(?:\s+\d+)*\n){7})',
        re.DOTALL
    )
    
    # First collect all data
    for match in case_pattern.finditer(content):
        case_type = match.group(1)
        pattern_text = match.group(2)
        
        pattern = []
        for line in pattern_text.strip().split('\n'):
            pattern.extend([int(x) for x in line.split()])
        pattern = np.array(pattern)
        
        expected_text = content[match.end():].split('\n')[0]
        expected_digit = int(re.search(r'Expected Digit: (\d+)', expected_text).group(1))
        
        output = simple_matrix_nn(pattern, weights1, weights2, [2, 2])
        predicted_digit = output.argmax() + 2 if sum(output) == 1 else None
        
        case_data = {
            'case_num': total_cases + 1,
            'pattern': pattern,
            'expected_digit': expected_digit,
            'predicted_digit': predicted_digit,
            'output': output
        }
        
        if case_type == "Correct" and (predicted_digit != expected_digit):
            cases_data['correct_failed'].append(case_data)
            correct_but_failed += 1
        elif case_type == "Incorrect" and predicted_digit == expected_digit:
            cases_data['incorrect_succeeded'].append(case_data)
            incorrect_but_succeeded += 1
        elif case_type == "Invalid" and predicted_digit == expected_digit:
            cases_data['invalid_succeeded'].append(case_data)
            invalid_but_succeeded += 1
            
        total_cases += 1
        if predicted_digit == expected_digit:
            correct_predictions += 1
    
    # Write verification differences file
    with open('verification_differences.txt', 'w') as df:
        df.write("=== Verification Differences Analysis ===\n\n")
        
        # Write statistics
        df.write("Statistics:\n")
        df.write("===========\n")
        df.write(f"Total Cases Analyzed: {total_cases}\n")
        df.write(f"Cases that should be correct but failed: {correct_but_failed}\n")
        df.write(f"Cases that were marked as incorrect but succeeded: {incorrect_but_succeeded}\n")
        df.write(f"Cases that were marked as invalid but succeeded: {invalid_but_succeeded}\n")
        df.write(f"Total differences: {correct_but_failed + incorrect_but_succeeded + invalid_but_succeeded}\n\n")
        
        # Write detailed cases
        if correct_but_failed > 0:
            df.write("1. Cases that should be correct but failed:\n")
            df.write("==========================================\n\n")
            for case in cases_data['correct_failed']:
                df.write(f"\nCase {case['case_num']} (Should be correct but failed):\n")
                df.write("Input Pattern:\n")
                for i in range(7):
                    df.write(" ".join(map(str, case['pattern'][i*7:(i+1)*7])) + "\n")
                df.write(f"Expected Digit: {case['expected_digit']}\n")
                df.write(f"Predicted Digit: {case['predicted_digit'] if case['predicted_digit'] is not None else 'Invalid'}\n")
                df.write(f"Expected Output: {[1.0 if i == case['expected_digit']-2 else 0.0 for i in range(3)]}\n")
                df.write(f"Actual Output: {case['output'].tolist()}\n")
                df.write("=" * 50 + "\n")
        
        if incorrect_but_succeeded > 0:
            df.write("\n2. Cases that were marked as incorrect but are now correct:\n")
            df.write("=========================================================\n\n")
            for case in cases_data['incorrect_succeeded']:
                df.write(f"\nCase {case['case_num']} (Marked as incorrect but predicted correctly):\n")
                df.write("Input Pattern:\n")
                for i in range(7):
                    df.write(" ".join(map(str, case['pattern'][i*7:(i+1)*7])) + "\n")
                df.write(f"Expected Digit: {case['expected_digit']}\n")
                df.write(f"Predicted Digit: {case['predicted_digit']}\n")
                df.write(f"Expected Output: {[1.0 if i == case['expected_digit']-2 else 0.0 for i in range(3)]}\n")
                df.write(f"Actual Output: {case['output'].tolist()}\n")
                df.write("=" * 50 + "\n")
        
        if invalid_but_succeeded > 0:
            df.write("\n3. Cases that were marked as invalid but are now correct:\n")
            df.write("=========================================================\n\n")
            for case in cases_data['invalid_succeeded']:
                df.write(f"\nCase {case['case_num']} (Marked as invalid but predicted correctly):\n")
                df.write("Input Pattern:\n")
                for i in range(7):
                    df.write(" ".join(map(str, case['pattern'][i*7:(i+1)*7])) + "\n")
                df.write(f"Expected Digit: {case['expected_digit']}\n")
                df.write(f"Predicted Digit: {case['predicted_digit']}\n")
                df.write(f"Expected Output: {[1.0 if i == case['expected_digit']-2 else 0.0 for i in range(3)]}\n")
                df.write(f"Actual Output: {case['output'].tolist()}\n")
                df.write("=" * 50 + "\n")
    
    # Write complete verification results
    with open('verification.txt', 'w') as vf:
        vf.write("=== Verification Results ===\n\n")
        vf.write(f"Total cases: {total_cases}\n")
        vf.write(f"Correct predictions: {correct_predictions}\n")
        vf.write(f"Incorrect predictions: {total_cases - correct_predictions}\n")
        vf.write(f"Accuracy: {100.0 * correct_predictions / total_cases:.2f}%\n\n")
        
        vf.write("Network Weights:\n")
        vf.write("Weights1:\n")
        for row in weights1:
            vf.write(" ".join(map(str, row)) + "\n")
        vf.write("\nWeights2:\n")
        for row in weights2:
            vf.write(" ".join(map(str, row)) + "\n")
    
    print(f"\nVerification results have been saved to 'verification.txt'")
    print(f"Differences analysis has been saved to 'verification_differences.txt'")
    print(f"\nDifferences Summary:")
    print(f"Cases that should be correct but failed: {correct_but_failed}")
    print(f"Cases that were marked as incorrect but are now correct: {incorrect_but_succeeded}")
    print(f"Cases that were marked as invalid but are now correct: {invalid_but_succeeded}")
    print(f"Total differences: {correct_but_failed + incorrect_but_succeeded + invalid_but_succeeded}")
    print(f"\nTotal cases: {total_cases}")
    print(f"Accuracy: {100.0 * correct_predictions / total_cases:.2f}%")

# Run the verification
verify_all_cases('correct_cases.txt', 'final_weights.txt')

