import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score
from gennet8 import QuantLayer
import torch.nn.functional as F  # 添加这行导入
from torch.optim.lr_scheduler import OneCycleLR  # 确保这行导入存在
class TrainingLogger:
    def __init__(self, log_file="training_results.txt", start_save_epoch=0):  # 添加 start_save_epoch 参数
        self.log_file = log_file
        self.weights_file = log_file.rsplit('.', 1)[0] + '_weights.txt'
        self.best_accuracy = 0.0
        self.best_weights = None
        self.start_save_epoch = start_save_epoch  # 添加开始保存的周期
        
    def log_batch(self, epoch, batch_idx, loss, accuracy, lr=None, ce_loss=None, reg_loss=None):
        """记录每个batch的训练信息"""
        message = f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%'
        if lr is not None:
            message += f', LR: {lr:.6f}'
        if ce_loss is not None and reg_loss is not None:
            message += f'\nCE Loss: {ce_loss:.4f}, Reg Loss: {reg_loss:.4f}'
        
        print(message)
        # 同时写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    def log_epoch(self, epoch, num_epochs, train_loss, train_accuracy, 
                  test_loss=None, test_accuracy=None, net=None, class_accuracies=None):
        """记录每个epoch的训练信息"""
        message = f'\nEpoch [{epoch}/{num_epochs}], '
        message += f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%'
        
        if test_loss is not None and test_accuracy is not None:
            message += f', Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%'
            
            # 添加每个类别的准确率信息
            if class_accuracies is not None:
                message += '\nPer-class Accuracies:'
                for class_idx, stats in class_accuracies.items():
                    message += f'\nClass {class_idx}: {stats["accuracy"]:.2f}% ({stats["correct"]}/{stats["total"]})'
            
            # 只有在达到指定epoch后才开始记录最佳权重
            if epoch >= self.start_save_epoch:
                if test_accuracy > self.best_accuracy:
                    self.best_accuracy = float(test_accuracy)
                    message += f'\nNew Best Accuracy: {self.best_accuracy:.2f}%'
                    
                    # 保存最佳权重到单独的文件
                    if net is not None:
                        self.save_best_weights(net)
        
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    def save_best_weights(self, net):
        """保存达到最佳准确率时的权重到单独的文件"""
        with open(self.weights_file, 'w', encoding='utf-8') as f:
            f.write(f"\nBest Model Weights (Accuracy: {self.best_accuracy:.2f}%):\n")
            f.write("=" * 50 + "\n")
            
            for i, layer in enumerate(net.layers):
                if isinstance(layer, QuantLayer):
                    # 先强制执行约束
                    layer.enforce_constraints()
                    weight = layer.quantize_weights().detach().cpu().numpy()
                    
                    f.write(f"\nLayer {i} Weights Matrix ({weight.shape[0]}x{weight.shape[1]}):\n")
                    f.write("=" * 50 + "\n")
                    
                    # 验证每列的约束
                    for col in range(weight.shape[1]):
                        column = weight[:, col]
                        pos_count = np.sum(column > 0)
                        neg_count = np.sum(column < 0)
                        if pos_count > 6 or neg_count > 2:
                            print(f"Warning: Layer {i}, Column {col} violates constraints!")
                            print(f"Positive weights: {pos_count}, Negative weights: {neg_count}")
                    
                    # 保存权重矩阵
                    np.savetxt(f, weight, fmt='%d')
                    
                    # 打印统计信息
                    total = weight.size
                    pos = np.sum(weight > 0)
                    neg = np.sum(weight < 0)
                    zero = np.sum(weight == 0)
                    f.write("\nMatrix Statistics:\n")
                    f.write(f"Total elements: {total}\n")
                    f.write(f"Number of 1s: {pos} ({pos/total*100:.2f}%)\n")
                    f.write(f"Number of -1s: {neg} ({neg/total*100:.2f}%)\n")
                    f.write(f"Number of 0s: {zero} ({zero/total*100:.2f}%)\n\n")
    
    def log_weights(self, net, epoch, first_write=False):
        """记录权分布"""
        mode = 'w' if first_write else 'a'
        # 使用 utf-8 编码写入文件
        with open(self.log_file, mode, encoding='utf-8') as f:
            f.write(f"\nEpoch {epoch} Layer Weights Summary:\n")
            f.write("-" * 50 + "\n")
            
            f.write("\nQuantization Distribution:\n")
            import io
            from contextlib import redirect_stdout
            s = io.StringIO()
            with redirect_stdout(s):
                monitor_quantization(net)
            f.write(s.getvalue())

def evaluate_spikes(spikes, targets, print_details=False):
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
                
        if print_details:
            print(f"\nSample {i}:")
            print(f"Spike values: {spike.tolist()}")
            print(f"Target: {targets[i].item()}")
            print(f"Sum of spikes: {spike.sum().item()}")
            print(f"Max spike value: {spike.max().item()}")
    
    return correct

def monitor_quantization(model):
    """Monitor weight distribution in quantized layers"""
    for i, layer in enumerate(model.layers):
        if isinstance(layer, QuantLayer):
            print(f"\nLayer {i} Statistics:")
            print(f"Quantization Status: {'Enabled' if layer.enable_quantization else 'Disabled'}")
            
            # Get weights (quantized or not)
            weight = layer.quantize_weights().detach()
            
            total = weight.numel()
            unique, counts = torch.unique(weight, return_counts=True)
            
            print(f"Weight Statistics ({weight.size(0)}x{weight.size(1)}):")
            print(f"Number of unique values: {len(unique)}")
            print(f"Value range: [{weight.min().item():.4f}, {weight.max().item():.4f}]")
            
            # If quantized, should only have three values
            if layer.enable_quantization:
                print("\nQuantized Value Distribution:")
                sorted_values = sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: abs(x[0]))
                for val, count in sorted_values:
                    print(f"Value {val:6.3f}: {count:6d} ({100.0 * count / total:6.2f}%)")
                
                # Print current threshold
                if hasattr(layer, 'threshold') and layer.threshold is not None:
                    print(f"Current threshold: {layer.threshold:.4f}")

def analyze_weights(net, epoch, first_write=False):
    """
    分析并记录网络权重分布
    - 将权重分析结果写入文件
    - first_write: 是否是第一次写入（决定是覆盖还是追加模式）
    """
    mode = 'w' if first_write else 'a'
    
    with open("training_results.txt", mode) as f:
        f.write(f"\nEpoch {epoch} Layer Weights Summary:\n")
        f.write("-" * 50 + "\n")
        
        f.write("\nQuantization Distribution:\n")
        import io
        from contextlib import redirect_stdout
        s = io.StringIO()
        with redirect_stdout(s):
            monitor_quantization(net)
        f.write(s.getvalue())

def create_optimizer_with_layer_lrs(net, optimizer_fn, normal_lr, quant_lr):
    """
    为不同类型的层创建不同学习率的优化器
    Args:
        net: 神经网络模型
        optimizer_fn: 优化器构造函数
        normal_lr: 普通层的学习率
        quant_lr: 量化层的学习率
    """
    param_groups = []
    
    for i, layer in enumerate(net.layers):
        if isinstance(layer, QuantLayer):
            param_groups.append({
                'params': layer.parameters(),
                'lr': quant_lr,
                'layer_idx': i,
                'layer_type': 'quant'
            })
            print(f"Layer {i} (QuantLayer): lr = {quant_lr}")
        else:
            if hasattr(layer, 'parameters'):
                param_groups.append({
                    'params': layer.parameters(),
                    'lr': normal_lr,
                    'layer_idx': i,
                    'layer_type': 'normal'
                })
                print(f"Layer {i} (Normal): lr = {normal_lr}")
    
    return optimizer_fn(param_groups)

def adjust_learning_rate_for_quantization(optimizer, net, epoch, factor=0.1):
    """当量化层开始量化时降低其学习率"""
    for i, layer in enumerate(net.layers):
        if isinstance(layer, QuantLayer):
            # 检查是否是该层刚开始量化
            if layer.enable_quantization and layer.epoch_counter == 0:
                # 找到对应的参数组并调整学习率
                for param_group in optimizer.param_groups:
                    if param_group['layer_type'] == 'quant' and param_group['layer_idx'] == i:
                        old_lr = param_group['lr']
                        param_group['lr'] *= factor  # 降低到原来的0.1倍
                        print(f"\nEpoch {epoch}: Layer {i} starts quantization.")
                        print(f"Reducing learning rate from {old_lr:.6f} to {param_group['lr']:.6f}")

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
       normal_lr=0.01,
       quant_lr=0.001,
       loss_fn=None,
       eval_interval=1,
       scheduler_config=None,
       start_save_epoch=0,
       max_constraint_weight=0.1,
):
    """
    训练循环的主要步骤：
    1. 训练阶段：
       - 重置隐藏状态
       - 前向传播
       - 计算损失
       - 反向传播
       - 更新权重
       - 记录训练指标
    
    2. 评估阶段（每个eval_interval轮进行一次）：
       - 在测试集上评估模型
       - 记录测试指标
       - 更新最佳准确率
       
    3. 日志记录：
       - 将训练和测试结果写入日志文件
       - 打印训练进度
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    # 创建优化器
    optimizer = create_optimizer_with_layer_lrs(
        net, 
        optimizer_fn, 
        normal_lr=normal_lr, 
        quant_lr=quant_lr
    )
    
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
        adjust_learning_rate_for_quantization(optimizer, net, epoch)
        constraint_weight = max_constraint_weight * (epoch / num_epochs)
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
            loss = loss_fn(spikes, mem_potentials, targets, net, constraint_weight)
            
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
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
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
                    loss = loss_fn(spikes, mem_potentials, targets, net, constraint_weight)
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

    print(f"训练完成. 最终最佳准确率: {logger.best_accuracy:.2f}%")
    return float(logger.best_accuracy)

class ProcessedDataLoader:
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
