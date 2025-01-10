import torch
import numpy as np
import snntorch as snn
from gennet8 import genet
from gennet8 import genet, QuantLayer  # 添加 QuantLayer 的导入
from indataprocess8 import MNISTDataProcessor
from verify import load_weights, simple_matrix_nn  # 改用正确的函数名

def load_weights_from_file(file_path, device):
    """从文件中加载权重"""
    weights = []
    current_matrix = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和非权重行
            if not line or line.startswith('=') or line.startswith('Best') or \
               line.startswith('Matrix') or line.startswith('Layer') or \
               line.startswith('Total') or line.startswith('Number'):
                # 当遇到新的矩阵标记时，保存当前矩阵
                if current_matrix and (line.startswith('==') or line.startswith('Matrix')):
                    weights.append(torch.tensor(current_matrix, device=device))
                    current_matrix = []
                continue
                
            # 确保行包含数字且不包含其他文本
            try:
                # 分割行并转换为浮点数列表
                row_values = [float(x) for x in line.split()]
                if row_values:  # 确保行不为空
                    current_matrix.append(row_values)
            except ValueError:
                continue  # 跳过无法转换为数字的行
    
    # 添加最后一个矩阵（如果存在）
    if current_matrix:
        weights.append(torch.tensor(current_matrix, device=device))
    
    return weights

def apply_weights_to_model(model, weights):
    """将权重应用到模型"""
    layer_idx = 0
    for layer in model.layers:
        if hasattr(layer, 'linear'):
            layer.linear.weight.data = weights[layer_idx]
            layer_idx += 1

def run_inference(model, data_loader, device):
    """运行推理并记录正确的分类结果"""
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
                # 注释掉所有中间结果打印
                #print(f"\nInference.py intermediate results for sample {i}:")
                for layer_idx, layer in enumerate(model.layers):
                    if isinstance(layer, QuantLayer):
                        single_input = layer(single_input)
                        #print(f"Layer {layer_idx} (Linear) output: {single_input.cpu().numpy()}")
                    elif isinstance(layer, snn.Leaky):
                        layer.mem = torch.zeros_like(layer.mem)
                        assert layer.reset_mechanism == 'zero'
                        #print(f"Layer {layer_idx} before forward:")
                        #print(f"  Membrane potential: {layer.mem}")
                        single_input, mem = layer(single_input)
                        #print(f"  After forward:")
                        #print(f"  Output spikes: {single_input}")
                        #print(f"  New membrane: {mem}")
                        #print(f"Layer {layer_idx} (Leaky) membrane potential: {mem.cpu().numpy()}")
                        #print(f"Layer {layer_idx} (Leaky) spikes: {single_input.cpu().numpy()}")
                
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
    
    # 注释掉测试统计信息的打印
    #print(f'\nTest Statistics:')
    #print(f'Total samples: {total}')
    #print(f'Correct classifications: {len(correct_cases)}')
    #print(f'Incorrect classifications: {len(incorrect_cases)}')
    #print(f'Invalid predictions: {len(invalid_cases)}')
    #print(f'Accuracy: {100.0 * len(correct_cases) / total:.2f}%')
    
    return correct_cases, incorrect_cases, invalid_cases

def save_classification_results(correct_cases, incorrect_cases, invalid_cases, total_samples):
    with open('correct_cases.txt', 'w') as f:
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Correct Classifications: {len(correct_cases)}\n")
        f.write(f"Incorrect Classifications: {len(incorrect_cases)}\n")
        f.write(f"Invalid Classifications: {len(invalid_cases)}\n")  # 添加无效案例数量
        f.write(f"Accuracy: {100.0 * len(correct_cases) / total_samples:.2f}%\n")
        f.write("=" * 50 + "\n\n")
        
        # 写入所有分类结果，包括无效案例
        for case_type, cases in [
            ("CORRECT CLASSIFICATIONS", correct_cases),
            ("INCORRECT CLASSIFICATIONS", incorrect_cases),
            ("INVALID CLASSIFICATIONS", invalid_cases)  # 添加无效案例
        ]:
            f.write(f"=== {case_type} (Total: {len(cases)}) ===\n")
            f.write("=" * 50 + "\n\n")
            
            for idx, case in enumerate(cases, 1):
                f.write(f"=== {case['type'].capitalize()} Classification Case {idx}/{len(cases)} ===\n")
                # 保存输入模式
                f.write("Input Pattern (7x7):\n")
                input_matrix = case['input'].reshape(7, 7)
                for row in input_matrix:
                    f.write(" ".join("1" if x > 0.5 else "0" for x in row) + "\n")
                
                # 保存目标数字和输出值
                f.write(f"Expected Digit: {case['target']}\n")  # 添加期望的数字
                if case['type'] != 'invalid':
                    f.write(f"Recognized Digit: {case['output'].argmax() + 2}\n")
                else:
                    f.write("Recognized Digit: Invalid (multiple or no spikes)\n")
                f.write(f"Output Spikes: {case['output'].tolist()}\n")
                f.write("="*50 + "\n")

def compare_results(inference_output, verify_output):
    """比较inference和verify的结果"""
    #print("\nComparing results:")
    #print(f"Inference output: {inference_output}")
    #print(f"Verify output: {verify_output}")
    #print(f"Match: {np.allclose(inference_output, verify_output)}")
    pass

def main():
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")
    
    # 模型参数（与训练时相同）
    num_inputs = 49
    num_hidden = [24]
    num_outputs = 3
    beta_values = [0,0]
    thresholds = [1.99, 1.99]
    
    # 创建模型
    model, _ = genet(num_inputs, num_hidden, num_outputs, beta_values, thresholds, 
                    optimizer_fn=None, quant_start_epoch=50)
    model = model.to(device)
    
    # 加载权重（确保权重在正确的设备上）
    weights = load_weights_from_file('final_weights.txt', device)
    apply_weights_to_model(model, weights)
    
    # 准备数据
    data_processor = MNISTDataProcessor(batch_size=64)
    _, test_loader = data_processor.get_processed_loaders()
    
    # 运行inference
    correct_cases, incorrect_cases, invalid_cases = run_inference(model, test_loader, device)
    
    # 运行verify进行比较
    weights1, weights2 = load_weights('final_weights.txt')
    
    # 对于每个样本进行比较
    with torch.no_grad():  # 添加这行
        for data, targets in test_loader:
            for i in range(len(data)):
                single_input = data[i].numpy().flatten()
                
                # 运行verify
                verify_output = simple_matrix_nn(single_input, weights1, weights2, thresholds)
                
                # 运行inference
                model.reset_hidden()
                inference_output = model(torch.tensor(single_input).float().to(device))[0]
                inference_output = inference_output.detach().cpu().numpy()  # 修改这行
                
                # 比较结果
                print("\nComparing results:")
                print(f"Inference output: {inference_output}")
                print(f"Verify output: {verify_output}")
                print(f"Match: {np.allclose(inference_output, verify_output)}")
                
                # 只比较几个样本就退出
                if i >= 5:
                    break
            break
    
    # 计算总样本数
    total_samples = sum(len(data) for data, _ in test_loader)
    
    # 保存分类案例和准确率
    save_classification_results(correct_cases, incorrect_cases, invalid_cases, total_samples)
    #print(f"\n已保存 {len(correct_cases)} 个正确分类案例、 {len(incorrect_cases)} 个错误分类案例和 {len(invalid_cases)} 个无效预测案例到 'correct_classifications' 文件夹")
    #print(f"准确率: {100.0 * len(correct_cases) / total_samples:.2f}%")

if __name__ == "__main__":
    main()