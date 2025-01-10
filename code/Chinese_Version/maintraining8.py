# maintraining_mnist.py
import torch
import torch.nn as nn
import warnings
import numpy as np
from indataprocess8 import MNISTDataProcessor
import snntorch as snn
from gennet8 import genet
from Trainloop8 import train_model, create_optimizer_with_layer_lrs
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F  
from gennet8 import QuantLayer

# GPU 可用性检查
#print("PyTorch版本:", torch.__version__)
#print("CUDA是否可用:", torch.cuda.is_available())
#if torch.cuda.is_available():
    #print("CUDA版本:", torch.version.cuda)
    #print("GPU设备数量:", torch.cuda.device_count())
    #print("当前GPU设备名称:", torch.cuda.get_device_name(0))
    #print("当前GPU设备索引:", torch.cuda.current_device())

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"\n使用设备: {device}")

# 过滤警告
#warnings.filterwarnings('ignore', category=UserWarning)

# 加载和处理数据集
batch_size = 128
data_processor = MNISTDataProcessor(batch_size=batch_size)
processed_train_loader, processed_test_loader = data_processor.get_processed_loaders()

# 初始化模型参数
num_inputs = 49  
num_hidden = [24]  
num_outputs = 3   
beta_values = [0,0]  
thresholds = [1.99, 1.99]  

# 定义量化开始的epoch
quant_start_epoch = 70 

# 定义每层的输入剪枝配置，现在使用字典来指定每层的保留连接数
pruning_config = {
    0: {
        'input_connections': 8,  # 严格限制为8个输入连接
        'start_epoch': 30
    },
    1: {  # 这里变成了输出层的配置（原来是中间层）
        'input_connections': 8,  # 限制每个输出神经元最多8个输入连接
        'start_epoch': 50
    }
}

# 自定义优化器
'''
optimizer_fn = lambda params: torch.optim.SGD(
    params,
    lr=0.02,
    momentum=0.9,
    weight_decay=0.0,  # 显式设置为0，效果和不写一样
    nesterov=True
)
'''
#adamW优化器
'''
optimizer_fn = lambda params: torch.optim.AdamW(
    params, # 要优化的网络参数
    lr=0.003,  # 稍微提高初始学习率
    weight_decay=0.0000, # L2正则化系数：用于防止过拟合，这里设为0表示不使用
    betas=(0.9, 0.999) , # 动量系数：用于计算梯度的移动平均
                        # 第一个值(0.8)控制一阶矩估计
                        # 第二个值(0.999)控制二阶矩估计
)
'''
optimizer_fn = lambda params: torch.optim.AdamW(
    params, # 要优化的网络参数
    lr=0.05,  # 稍微提高初始学习率,之前是0.005
    weight_decay=0, # L2正则化系数：用于防止过拟合，这里设为0表示不使用,之前是0.0001
    betas=(0.9, 0.999) , # 动量系数：用于计算梯度的移动平均
                        # 第一个值(0.8)控制一阶矩估计
                        # 第二个值(0.999)控制二阶矩估计
)

'''正则化
def loss_fn_with_regularization(spikes, targets, model, epoch, lambda_reg_base=1e-4):
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)(spikes, targets)
    reg_loss = torch.tensor(0.0, device=ce_loss.device)
    
    # 获取所有 QuantLayer 的 quant_start_epoch
    quant_start_epochs = []
    for layer in model.layers:
        if isinstance(layer, QuantLayer):
            quant_start_epochs.append(layer.quant_start_epoch)
            if layer.threshold is not None:
                quantized_weight = layer.ternary_quant(layer.linear.weight, layer.threshold)
                # 正则化项归一化
                reg_loss += ((layer.linear.weight - quantized_weight) ** 2).sum() / layer.linear.weight.numel()
    
    # 计算最小的 quant_start_epoch
    min_quant_start_epoch = min(quant_start_epochs) if quant_start_epochs else None
    
    # 判断是否需要增大正则化系数
    max_lambda_reg = 0.001  # 设置正则化系数的最大值
    if min_quant_start_epoch is not None and epoch >= min_quant_start_epoch:
        epochs_since_quant = epoch - min_quant_start_epoch + 1
        lambda_reg = min(lambda_reg_base * np.log1p(epochs_since_quant), max_lambda_reg)
    else:
        lambda_reg = lambda_reg_base
    
    total_loss = ce_loss + lambda_reg * reg_loss
    return total_loss, ce_loss.item(), (lambda_reg * reg_loss).item()
'''
def combined_loss_with_constraints(spikes, mem_potentials, targets):
    """
    组合损失函数，同时考虑脉冲、膜电位和权重约束
    Args:
        spikes: 网络输出的脉冲
        mem_potentials: 网络输出的膜电位
        targets: 目标标签
        model: 网络模型
        constraint_weight: 约束的权重系数
    Returns:
        组合后的损失值
    """
    spike_loss = nn.CrossEntropyLoss()(spikes, targets)
    norm_mem = F.softmax(mem_potentials, dim=1)
    mem_loss = nn.CrossEntropyLoss()(norm_mem, targets)

    total_loss = 0.6 * spike_loss + 0.4 * mem_loss
    return total_loss   
#loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)  # 移除标签平滑

# 训练参数
num_epochs = 300
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n使用设备: {device}")

# 修改模型创建
model, forward_fn = genet(
    num_inputs=num_inputs,
    num_hidden=num_hidden,
    num_outputs=num_outputs,
    beta_values=beta_values,
    thresholds=thresholds,
    optimizer_fn=optimizer_fn,
    pruning_config=pruning_config,
    quant_start_epoch=quant_start_epoch
)

# 创建优化器
optimizer = create_optimizer_with_layer_lrs(
    model, 
    optimizer_fn, 
)

# 添加scheduler配置
scheduler_config = {
    'enabled': False,  # 是否启用scheduler
    'type': 'onecycle',  # scheduler类型
    'params': {
        'max_lr': 0.02,
        'div_factor': 10,
        'pct_start': 0.3,
    }
}

# 修改scheduler创建部分
if scheduler_config['enabled']:
    if scheduler_config['type'] == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            **scheduler_config['params'],
            epochs=num_epochs,
            steps_per_epoch=len(processed_train_loader),
        )
    else:
        scheduler = None
else:
    scheduler = None

# 打印模型结构和参数数量
#print("模型结构:")
#print(model)
#print("\n模型参数:")
#total_params = sum(p.numel() for p in model.parameters())
#print(f"总参数数量: {total_params:,}")

# 打印剪枝配置信息
#print("\n剪枝配置:")
#layer_sizes = [num_inputs] + num_hidden + [num_outputs]
#for i in range(len(layer_sizes)-1):
    #prune_config = pruning_config.get(i)
    #if prune_config is None:
        #print(f"第{i+1}层 ({layer_sizes[i]}->{layer_sizes[i+1]}): 不剪枝")
        
    #else:
        #print(f"第{i+1}层 ({layer_sizes[i]}->{layer_sizes[i+1]}): "
              #f"每个神经元保留{prune_config['input_connections']}个输入连接, "
              #f"开始于epoch {prune_config['start_epoch']}")
        

# 训练模型
best_accuracy = train_model(
    net=model,
    forward_fn=forward_fn,
    train_loader=processed_train_loader,
    test_loader=processed_test_loader,
    num_epochs=num_epochs,
    start_save_epoch=101,
    device=device,
    loss_fn=combined_loss_with_constraints,
    eval_interval=1,
    scheduler_config=scheduler_config,
    optimizer_fn=optimizer_fn,
)

# 移除 "训练完成。" 的打印