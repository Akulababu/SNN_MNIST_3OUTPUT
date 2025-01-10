import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

class TernaryQuantFunction(torch.autograd.Function):
    """三值量化函数，使用Straight-Through Estimator进行反向传播"""
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        output = torch.zeros_like(input)
        output[input > threshold] = 1.0    # 大于阈值的为1
        output[input < -threshold] = -1.0  # 小于负阈值的为-1
        return output

    @staticmethod
    # 反向传播函数，用了C++的SpikeTorch库,不容易debug
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None   

class QuantLayer(nn.Module):
    def __init__(self, in_features, out_features, prune_config=None, quant_start_epoch=10, layer_index=0, threshold_method='scaled_std'):
        super().__init__()
        
        # 初始化设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 普通线性层
        self.linear = nn.Linear(in_features, out_features, bias=False).to(self.device)
        
        # 掩码初始化
        self.mask = torch.ones(out_features, in_features).to(self.device)
        
        # 量化控制参数
        self.enable_quantization = False
        self.quant_start_epoch = quant_start_epoch
        
        self.epoch_counter = 0
        self.pruning_counter = 0
        self.prune_config = prune_config
        
        # 动态阈值参数
        self.threshold = None  # 将在量化时动态计算
        self.threshold_method = threshold_method
        
        # 保存层的大小信息
        self.in_features = in_features
        self.out_features = out_features
        
        # 三值量化函数
        self.ternary_quant = TernaryQuantFunction.apply
        
        #self.quant_start_epoch = quant_start_epoch + layer_index * delay_between_layers
        
        # 在 QuantLayer 类中，添加一个变量来保存固定的阈值
        self.fixed_threshold = None
        
        # 根据层索引设置不同的阈值计算方法
        self.layer_index = layer_index
        self.threshold_method = threshold_method
        
        # 添加权重约束相关的属性
        self.max_positive_weights = 6  # 每个输出神经元最多可以有6个正权重
        self.max_negative_weights = 2  # 每个输出神经元最多可以有2个负权重
        
        # 使用 Kaiming 初始化替换原来的 uniform 初始化
        nn.init.kaiming_normal_(self.linear.weight, 
                              mode='fan_in',  # 使用fan_in模式
                              nonlinearity='leaky_relu',  # 使用leaky_relu更适合SNN
                              a=0.1)  # 较小的slope参数
        
        # 添加权重范围限制
        with torch.no_grad():
            self.linear.weight.data.clamp_(-1, 1)
        
        # 添加标准化相关的属性
        self.weight_mean = None
        self.weight_std = None
        self.is_normalized = False
        
    def calculate_threshold(self, weights):
        """根据层的位置使用不同的阈值计算方法"""
        if self.layer_index == 0:  # 第一层
            abs_weights = weights.abs()
            
            # 方法1: 基于百分位数
            #threshold1 = torch.quantile(abs_weights, 0.93)
            
            # 方法2: 基于标准差
            #mean = abs_weights.mean()
            #std = abs_weights.std()
           #threshold2 = mean + 1.5 * std
            
            # 方法3: 使用topk
            #total_weights = weights.numel()
            #keep_weights = int(total_weights * 0.07)
            #threshold3 = torch.topk(abs_weights.view(-1), keep_weights, largest=True)[0][-1]
            
            #方法4: 基于权重分布的自适应阈值
            sorted_weights, _ = torch.sort(abs_weights.view(-1))
            target_idx = int(len(sorted_weights) * 0.93)
            window_size = len(sorted_weights) // 20
            start_idx = max(0, target_idx - window_size // 2)
            end_idx = min(len(sorted_weights), target_idx + window_size // 2)
            threshold4 = sorted_weights[start_idx:end_idx].mean()
            
            # 方法5: 基于直方图的方法 (修改为CPU版本)
            #abs_weights_cpu = abs_weights.cpu()  # 移到CPU上计算
           # hist = torch.histogram(abs_weights_cpu, bins=100)
            #cumsum = torch.cumsum(hist.hist, 0)
           # target_sum = int(cumsum[-1] * 0.93)  # 目标93%的累积和
            # 找到最接近目标值的bin
            #idx = torch.searchsorted(cumsum, target_sum)
            #threshold5 = hist.bin_edges[idx].to(weights.device)  # 将结果移回原设备

            # 计算每种方法的实际稀疏度
            #sparsity1 = (abs_weights < threshold1).float().mean().item()
            #sparsity2 = (abs_weights < threshold2).float().mean().item()
            #sparsity3 = (abs_weights < threshold3).float().mean().item()
            #sparsity4 = (abs_weights < threshold4).float().mean().item()
            #sparsity5 = (abs_weights < threshold5).float().mean().item()

            # 打印稀疏度信息（可选）
            #if self.epoch_counter % 10 == 0:  # 每10个epoch打印一次
                #print(f"\nLayer {self.layer_index} Sparsity:")
                #print(f"Method 1 (Percentile): {sparsity1:.3f}")
                #print(f"Method 2 (Std): {sparsity2:.3f}")
                #print(f"Method 3 (TopK): {sparsity3:.3f}")
                #print(f"Method 4 (Adaptive): {sparsity4:.3f}")
                #print(f"Method 5 (Histogram): {sparsity5:.3f}")

            # 默认使用方法1，您可以根据实验结果更改使用的方法
            return threshold4
            
        #elif self.layer_index == 1:  # 中间层
            # 使用中等阈值
            #mean = weights.abs().mean()
            #std = weights.abs().std()
            #return 0.3 * mean + 0.2 * std
            
        else:  # 最后一层
            # 使用较小的阈值，保持更多的连接
            mean = weights.abs().mean()
            std = weights.abs().std()
            return 0.1 * mean + 0.05 * std
    
    def print_weight_stats(self, w, is_quantized=False):
        """打印权重统计信息 - 现在被注释"""
        pass  # 移除所有打印语句
        
    def to(self, device):
        """确保所有组件都移动到正确的设备"""
        super().to(device)
        self.device = device
        self.linear = self.linear.to(device)
        self.mask = self.mask.to(device)
        return self
    
    def enforce_constraints(self):
        with torch.no_grad():
            weights = self.linear.weight.data

            for i in range(self.out_features):  # 遍历每一行
                row = weights[i, :]
                mask_row = self.mask[i, :]  # 获取该行的剪枝掩码

                # 只考虑未被剪枝的权重
                valid_weights = row * mask_row

                # 处理正权重
                positive_mask = (valid_weights > 0)
                positive_weights = valid_weights[positive_mask]
                if len(positive_weights) > self.max_positive_weights:
                    # 保留最大的 max_positive_weights 个正权重
                    values, indices = torch.topk(positive_weights, self.max_positive_weights)

                    # 创建新的权重向量，初始化为0
                    new_row = torch.zeros_like(row)

                    # 获取原始正权重的索引，并只选择未被剪枝的
                    positive_indices = torch.nonzero(positive_mask & (mask_row == 1)).squeeze()
                    selected_indices = positive_indices[indices]
                    new_row[selected_indices] = values

                    # 保持负权重不变，但只保留未被剪枝的
                    negative_mask = (valid_weights < 0)
                    new_row[negative_mask] = valid_weights[negative_mask]

                    # 更新行
                    weights[i, :] = new_row

                # 处理负权重
                negative_mask = (valid_weights < 0)
                negative_weights = valid_weights[negative_mask]
                if len(negative_weights) > self.max_negative_weights:
                    # 保留绝对值最大的 max_negative_weights 个负权重
                    values, indices = torch.topk(negative_weights.abs(), self.max_negative_weights)
                    negative_indices = torch.nonzero(negative_mask & (mask_row == 1)).squeeze()
                    selected_indices = negative_indices[indices]

                    # 创建新的权重向量，保持正权重不变
                    temp_row = weights[i, :].clone()
                    temp_row[negative_mask] = 0  # 清除所有负权重
                    temp_row[selected_indices] = negative_weights[indices]  # 只保留选中的负权重

                    # 确保只更新未被剪枝的权重
                    weights[i, :] = temp_row * mask_row

            # 确保权重矩阵更新后满足约束
            self.linear.weight.data = torch.clamp(weights, -1.0, 1.0)

    def normalize_weights(self):
        """对权重进行标准化处理并限制在[-1,1]范围内"""
        with torch.no_grad():
            weights = self.linear.weight.data
            self.weight_mean = weights.mean()
            self.weight_std = weights.std()
            normalized_weights = (weights - self.weight_mean) / (self.weight_std + 1e-8)
            normalized_weights = torch.clamp(normalized_weights, -1.0, 1.0)
            self.linear.weight.data = normalized_weights
            self.is_normalized = True       
    '''
    def denormalize_weights(self):
        """将权重反标准化回原始尺度"""
        if self.is_normalized and self.weight_mean is not None and self.weight_std is not None:
            with torch.no_grad():
                weights = self.linear.weight.data
                denormalized_weights = weights * self.weight_std + self.weight_mean
                self.linear.weight.data = denormalized_weights
                self.is_normalized = False
    '''
    def forward(self, x):
        # 始终进行标准化
        if not self.is_normalized:
            self.normalize_weights()
        
        # 首先检查是否需要进行剪枝
        if (self.prune_config is not None and 
            self.epoch_counter >= self.prune_config['start_epoch']):
            self.update_mask()
        
        # 始终使用量化后的权重进行前向传播
        self.enforce_constraints()
        weights = self.linear.weight * self.mask
        if self.enable_quantization:
            # 在量化前确保权重已标准化
            if not self.is_normalized:
                self.normalize_weights()
            self.threshold = self.calculate_threshold(weights)
            w_q = self.ternary_quant(weights, self.threshold)
        else:
            # 即使在非量化模式下，也确保权重在[-1,1]范围内
            w_q = torch.clamp(weights, -1.0, 1.0)
        
        return F.linear(x, w_q)

    def update_mask(self):
        """更新剪枝掩码"""
        if self.prune_config is None:
            return
        
        with torch.no_grad():
            weights = self.linear.weight.data.abs()  # 使用绝对值
            new_mask = torch.zeros_like(self.mask, device=self.device)
            
            # 对每个输出神经元进行剪枝
            for i in range(self.out_features):
                # 获取该神经元的所有输入权重的绝对值
                w = weights[i]
                # 保留最大的k个权重
                k = self.prune_config['input_connections']
                # 找到最大的k个权重的索引
                _, indices = torch.topk(w, k)
                # 在掩码中将这些位置设为1
                new_mask[i][indices] = 1
            
            # 更新掩码并应用
            self.mask = new_mask
            self.linear.weight.data *= self.mask

    def on_epoch_end(self):
        """在每个epoch结束时调用"""
        self.epoch_counter += 1
        
        if self.epoch_counter == self.quant_start_epoch - 1:
            # print(f"\n准备开始量化 - Layer {self.out_features}x{self.in_features}")  # 注释掉
            self.normalize_weights()
        
        if (self.prune_config is not None and 
            self.epoch_counter == self.prune_config['start_epoch']):
            # print(f"\n开始剪枝 - Layer {self.out_features}x{self.in_features}")  # 注释掉
            self.update_mask()
            w = self.linear.weight.data * self.mask
            self.print_weight_stats(w, is_quantized=False)
        
        if not self.enable_quantization and self.epoch_counter >= self.quant_start_epoch:
            # print(f"\n开启量化 - Layer {self.out_features}x{self.in_features}")  # 注释掉
            self.enable_quantization = True
            w = self.linear.weight.data * self.mask
            self.print_weight_stats(w, is_quantized=False)
            self.threshold = self.calculate_threshold(w)
            w_q = self.ternary_quant(w, self.threshold)
            self.linear.weight.data = w_q
            self.print_weight_stats(w_q, is_quantized=True)

    def quantize_weights(self):
        """返回当前的权重状态（已剪枝和/或已量化）"""
        with torch.no_grad():
            if self.enable_quantization:
                # 确保权重已标准化
                if not self.is_normalized:
                    self.normalize_weights()
                self.enforce_constraints()
                # 再进行量化
                self.threshold = self.calculate_threshold(self.linear.weight.data)
                pruned_weights = self.linear.weight * self.mask
                # 进行量化
                quantized_weights = self.ternary_quant(pruned_weights, self.threshold)
                # 将量化后的权重应用到模型中
                self.linear.weight.data.copy_(quantized_weights)
                return self.linear.weight.data
            else:
                # 如果未量化，返回已剪枝且满足约束的权重
                self.enforce_constraints()
                return self.linear.weight * self.mask

def genet(num_inputs, num_hidden, num_outputs, beta_values, thresholds, 
          optimizer_fn, pruning_config=None, quant_start_epoch=10):
    """
    创建和配置SNN网络
    
    Args:
        num_inputs: 输入特征数
        num_hidden: 隐藏层神经元数列
        num_outputs: 输出类别数
        beta_values: 各层的beta值列表
        thresholds: 各层的阈值列表
        optimizer_fn: 优化器函数
        pruning_config: 剪枝配置
        quant_start_epoch: 开始量化的epoch
    """
    spike_grad = surrogate.fast_sigmoid(slope=20)
    
    # 如果没有提供剪枝配置，创建默认配置（全部不剪枝）
    if pruning_config is None:
        pruning_config = {i: None for i in range(len(num_hidden) + 1)}
    
    # 创建网络
    net = QuantSNNNetwork(
        num_inputs=num_inputs,
        num_hidden=num_hidden,
        num_outputs=num_outputs,
        beta=beta_values,
        threshold=thresholds,
        pruning_config=pruning_config,
        quant_start_epoch=quant_start_epoch
    )
    
    # 单步前向传播
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
        
        # 输入层 -> 第一个隐藏层
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

        # 第一个隐藏层 -> 输出层
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
        """重置所有层的隐藏状态"""  
        for layer in self.layers:  
            if isinstance(layer, snn.Leaky):  
                # 如果 mem 已经存在，使用其形状创建新的零张量  
                if hasattr(layer, 'mem') and layer.mem is not None:  
                    layer.mem = torch.zeros_like(layer.mem)  
                if hasattr(layer, 'spk') and layer.spk is not None:  
                    layer.spk = torch.zeros_like(layer.spk)  