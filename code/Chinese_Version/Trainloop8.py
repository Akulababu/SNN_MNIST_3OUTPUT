import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score
from gennet8 import QuantLayer
import torch.nn.functional as F  
from torch.optim.lr_scheduler import OneCycleLR 
class TrainingLogger:
    def __init__(self, start_save_epoch=0):
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.start_save_epoch = start_save_epoch

    def log_batch(self, epoch, batch_idx, loss, accuracy, lr=None, ce_loss=None, reg_loss=None):
        pass  # 移除所有打印和写入操作

    def log_epoch(self, epoch, num_epochs, train_loss, train_accuracy, 
                  test_loss=None, test_accuracy=None, net=None, class_accuracies=None):
        """记录每个epoch的训练信息"""
        # 打印当前epoch的训练和测试准确率
        print(f"Epoch [{epoch}/{num_epochs}] - Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")
        
        # 更新最佳准确率，只在start_save_epoch之后保存
        if test_accuracy is not None and epoch >= self.start_save_epoch:
            if test_accuracy > self.best_accuracy:
                self.best_accuracy = float(test_accuracy)
                self.best_epoch = epoch
                print(f"New Best Accuracy: {self.best_accuracy:.2f}% at epoch {self.best_epoch}")
                # 保存最佳权重到文件
                self.save_best_weights(net)

    def save_best_weights(self, net):
        """保存最佳权重到文件"""
        with open("training_results_weights.txt", 'w') as f:
            f.write(f"\nBest Model Weights (Accuracy: {self.best_accuracy:.2f}%):\n")
            f.write("=" * 50 + "\n\n")
            
            for i, layer in enumerate(net.layers):
                if isinstance(layer, QuantLayer):
                    # 获取量化后的权重，而不是直接获取原始权重
                    weights = layer.quantize_weights()  # 使用quantize_weights方法获取当前的量化权重
                    f.write(f"Layer {i} Weights Matrix ({weights.size(0)}x{weights.size(1)}):\n")
                    f.write("=" * 50 + "\n")
                    
                    # 打印权重矩阵
                    for row in weights:
                        row_str = ' '.join([f"{x:g}" for x in row])
                        f.write(row_str + '\n')
                    
                    f.write('\n')
                    
                    # 计算统计信息
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
        pass  # 移除所有记录操作

def evaluate_spikes(spikes, targets):
    """评估神经网络输出的脉冲是否正确
    Args:
        spikes: 网络输出的脉冲值 [batch_size, 3]
        targets: 目标标签 [batch_size]
        print_details: 是否打印详细信息
    Returns:
        correct: 正确预测的数量
    """
    batch_size = spikes.size(0)
    correct = 0
    
    for i in range(batch_size):
        spike = spikes[i]
        # 检查是否只有一个神经元发放
        if spike.sum() == 1 and spike.max() == 1:
            pred = spike.argmax().item()
            if pred == targets[i].item():
                correct += 1
                
        #if print_details:
            #print(f"\nSample {i}:")
            #print(f"Spike values: {spike.tolist()}")
            #print(f"Target: {targets[i].item()}")
            #print(f"Sum of spikes: {spike.sum().item()}")
            #print(f"Max spike value: {spike.max().item()}")
    
    return correct

#def monitor_quantization(model):
    """Monitor weight distribution in quantized layers"""
    #for i, layer in enumerate(model.layers):
        #if isinstance(layer, QuantLayer):
            #print(f"\nLayer {i} Statistics:")
            #print(f"Quantization Status: {'Enabled' if layer.enable_quantization else 'Disabled'}")
            
            # Get weights (quantized or not)
            #weight = layer.quantize_weights().detach()
            
            #total = weight.numel()
            #unique, counts = torch.unique(weight, return_counts=True)
            
            #print(f"Weight Statistics ({weight.size(0)}x{weight.size(1)}):")
            #print(f"Number of unique values: {len(unique)}")
            #print(f"Value range: [{weight.min().item():.4f}, {weight.max().item():.4f}]")
            
            # If quantized, should only have three values
            #if layer.enable_quantization:
                #print("\nQuantized Value Distribution:")
                #sorted_values = sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: abs(x[0]))
                #for val, count in sorted_values:
                    #print(f"Value {val:6.3f}: {count:6d} ({100.0 * count / total:6.2f}%)")
                   
                
                # Print current threshold
                #if hasattr(layer, 'threshold') and layer.threshold is not None:
                    #print(f"Current threshold: {layer.threshold:.4f}")
                   
#def analyze_weights(net, epoch, first_write=False):
    """
    分析并记录网络权重分布
    - 将权重分析结果写入文件
    - first_write: 是否是第一次写入（决定是覆盖还是追加模式）
    """
    #mode = 'w' if first_write else 'a'
    
    #with open("training_results.txt", mode) as f:
        #f.write(f"\nEpoch {epoch} Layer Weights Summary:\n")
        #f.write("-" * 50 + "\n")
        
        #f.write("\nQuantization Distribution:\n")
        #import io
        #from contextlib import redirect_stdout
        #s = io.StringIO()
        #with redirect_stdout(s):
            #monitor_quantization(net)
        #f.write(s.getvalue())

def create_optimizer_with_layer_lrs(net, optimizer_fn):
    """
    为网络创建优化器，所有层使用相同的学习率
    Args:
        net: 神经网络模型
        optimizer_fn: 优化器构造函数
    """
    params = []
    
    for layer in net.layers:
        if hasattr(layer, 'parameters'):
            params.extend(layer.parameters())
    
    return optimizer_fn(params)

#def adjust_learning_rate_for_quantization(optimizer, net, epoch, factor=0.1):
    """当量化层开始量化时降低其学习率"""
    #for i, layer in enumerate(net.layers):
        #if isinstance(layer, QuantLayer):
            # 检查是否是该层刚开始量化
            #if layer.enable_quantization and layer.epoch_counter == 0:
                # 找到对应的参数组并调整学习率
                #for param_group in optimizer.param_groups:
                    #if param_group['layer_type'] == 'quant' and param_group['layer_idx'] == i:
                        #old_lr = param_group['lr']
                        #param_group['lr'] *= factor  # 降低到原来的0.1倍
                        #print(f"\nEpoch {epoch}: Layer {i} starts quantization.")
                        #print(f"Reducing learning rate from {old_lr:.6f} to {param_group['lr']:.6f}")

def evaluate_spikes_by_class(spikes, targets, num_classes=5):
    """
    按类别评估神经网络输出的脉冲准确率
    Args:
        spikes: 网络输出的脉冲
        targets: 目标标签
        num_classes: 类别数量
    Returns:
        dict: 每个类别的准确率统计
    """
    max_spike_indices = spikes.argmax(dim=1)
    class_stats = {i: {'correct': 0, 'total': 0} for i in range(num_classes)}
    
    # 统计每个类别的正确数和总数
    for pred, target in zip(max_spike_indices, targets):
        target_class = target.item()
        class_stats[target_class]['total'] += 1
        if pred == target:
            class_stats[target_class]['correct'] += 1
    
    # 计算每个类别的准确率
    class_accuracies = {}
    for class_idx, stats in class_stats.items():
        if stats['total'] > 0:
            accuracy = 100.0 * stats['correct'] / stats['total']
            class_accuracies[class_idx] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total']
            }
    
    return class_accuracies

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
    """
    训练循环的主要步骤
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    # 创建优化器，使用统一的学习率
    optimizer = create_optimizer_with_layer_lrs(net, optimizer_fn)
    
    # 根据配置创建 scheduler
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
        # 在每个epoch开始时检查是否需要调整学习率
        #adjust_learning_rate_for_quantization(optimizer, net, epoch)
        net.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        logger.log_weights(net, epoch + 1, first_write=(epoch == 0))
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device).requires_grad_(True)
            targets = targets.to(device)
            
            #net.reset_hidden()
            optimizer.zero_grad()
            
            spikes, mem_potentials = forward_fn(data)
            
            # 修改这里：传入 net 和 constraint_weight
            loss = loss_fn(spikes, mem_potentials, targets)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 如果启用了scheduler并且是OneCycleLR，在每个batch后更新
            if scheduler and scheduler_config['type'] == 'onecycle':
                scheduler.step()

            # 计算训练准确率
            train_total += targets.size(0)
            train_correct += evaluate_spikes(spikes, targets)
            train_loss += loss.item()

            if batch_idx % 10 == 0:
                pass
        
        # 在每个epoch结束时更新所有QuantLayer的epoch计数
        for layer in net.layers:
            if isinstance(layer, QuantLayer) and hasattr(layer, 'on_epoch_end'):
                layer.on_epoch_end()

        # 如果启用了其他类型的scheduler，在每个epoch后更新
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
                    
                    #net.reset_hidden()
                    spikes, mem_potentials = forward_fn(data)
                    
                    # 这里也需要修改：传入 net 和 constraint_weight
                    loss = loss_fn(spikes, mem_potentials, targets)
                    test_loss += loss.item()
                    
                    test_total += targets.size(0)
                    test_correct += evaluate_spikes(spikes, targets)
                    
                    # 收集所有预测结果用于计算每个类别的准确率
                    all_spikes.append(spikes)
                    all_targets.append(targets)
            
            # 合并所有批次的结果
            all_spikes = torch.cat(all_spikes, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # 计算每个类别的准确率
            class_accuracies = evaluate_spikes_by_class(all_spikes, all_targets)
            
            test_loss /= len(test_loader)
            test_accuracy = 100. * test_correct / test_total
            train_accuracy = 100. * train_correct / train_total
            
            logger.log_epoch(
                epoch + 1, 
                num_epochs,
                train_loss/len(train_loader),
                train_accuracy,
                test_loss,
                test_accuracy,
                net=net,
                class_accuracies=class_accuracies
            )

    # 训练结束后打印最终结果
    print(f"\nTraining completed. Final Best Accuracy: {logger.best_accuracy:.2f}% at epoch {logger.best_epoch}")
    
    return float(logger.best_accuracy)

class ProcessedDataLoader:
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
