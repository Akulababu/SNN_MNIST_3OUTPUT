import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import cv2
import random

class MNISTDataProcessor:
    """
    MNIST数据集加载和预处理类
    1. 使用OpenCV将28x28图像缩放为14x14
    2. 将图像二值化（转换为0和1）
    """
    def __init__(self, batch_size=64):
        """
        初始化数据处理器
        Args:
            batch_size: 批次大小
        """
        self.batch_size = batch_size
        
        # 加载MNIST数据集
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        
        # 筛选数字2-4的样本
        train_indices = []
        test_indices = []
        
        # 统计训练集中每个类别的样本数量
        class_counts = {i: 0 for i in range(2, 5)}  # 只记录2,3,4的样本数量
        for i, (_, label) in enumerate(train_dataset):
            if 1 < label < 5:  # 修改条件为只选择2,3,4
                class_counts[label] += 1
        
        # 找出最小的类别数量
        min_samples = min(class_counts.values())
        
        # 为每个类别创建单独的索引列表
        train_class_indices = {i: [] for i in range(2, 5)}  # 改为2,3,4
        for i, (_, label) in enumerate(train_dataset):
            if 1 < label < 5:  # 修改条件为只选择2,3,4
                train_class_indices[label].append(i)
        
        # 从每个类别中随机采样相同数量的样本
        for label in range(2, 5):  # 改为2,3,4
            selected_indices = random.sample(train_class_indices[label], min_samples)
            train_indices.extend(selected_indices)
            # 调整标签值（2,3,4 变为 0,1,2）
            for idx in selected_indices:
                train_dataset.targets[idx] = label - 2  # 修改标签映射
        
        # 对测试集做同样的处理
        test_class_indices = {i: [] for i in range(2, 5)}  # 改为2,3,4
        for i, (_, label) in enumerate(test_dataset):
            if 1 < label < 5:  # 修改条件为只选择2,3,4
                test_class_indices[label].append(i)
        
        # 找出测试集中最小的类别数量
        test_min_samples = min(len(indices) for indices in test_class_indices.values())
        
        # 从测试集的每个类别中随机采样相同数量的样本
        for label in range(2, 5):
            selected_indices = random.sample(test_class_indices[label], test_min_samples)
            test_indices.extend(selected_indices)
            # 调整标签值（2,3,4 变为 0,1,2）
            for idx in selected_indices:
                test_dataset.targets[idx] = label - 2
        
        # 打印数据集统计信息
        # print("\nDataset Statistics:")
        # print(f"Samples per class: {min_samples}")
        # print(f"Total train samples: {len(train_indices)}")
        # print(f"Total test samples: {len(test_indices)}")
        
        # 创建数据加载器
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
    
    @staticmethod
    def preprocess_batch(batch): 
        """
        预处理MNIST批次数据
        1. 使用OpenCV将28x28图像缩放为14x14
        2. 二值化处理
        
        Args:
            batch: 包含图像和标签的批次数据
            
        Returns:
            tuple: (处理后的图像数据, 标签)
        """
        images, labels = batch
        batch_size = images.size(0)
    
        images_np = images.squeeze().numpy()
        processed_images = np.zeros((batch_size, 7, 7), dtype=np.float32)
        
        # 使用固定阈值 0.3（建议从这个值开始尝试）
        threshold = 0.3  
        
        for i in range(batch_size):
            img = images_np[i]
            # 使用INTER_AREA进行降采样，保持笔画的连续性
            resized = cv2.resize(img, (7, 7), interpolation=cv2.INTER_AREA)
            binary = (resized > threshold).astype(np.float32)
            processed_images[i] = binary
        
             # 每100个batch检查一次激活率（调试用）
            if i % 100 == 0:
                activation_rate = binary.mean()
                # print(f"Batch {i} activation rate: {activation_rate:.3f}")  # 注释掉
    
        return torch.from_numpy(processed_images).view(batch_size, -1), labels
    
    def get_processed_loaders(self):
        """
        返回处理后的数据加载器
        
        Returns:
            tuple: (处理后的训练数据加载器, 处理后的测试数据加载器)
        """
        class ProcessedDataLoader:
            def __init__(self, original_loader):
                self.loader = original_loader
                self.batch_size = original_loader.batch_size
            
            def __iter__(self):
                for batch in self.loader:
                    yield MNISTDataProcessor.preprocess_batch(batch)
                    
            def __len__(self):
                return len(self.loader)
        
        return (ProcessedDataLoader(self.train_loader),
                ProcessedDataLoader(self.test_loader))
    
    def print_batch_statistics(self, batch_data, batch_labels):
        """
        打印批次数据的统计信息
        """
        pass  # 移除所有打印语句

def verify_preprocessing(processor):
    # 获取一个batch的数据
    for batch, labels in processor.train_loader:
        processed, _ = processor.preprocess_batch((batch, labels))
        # 注释掉所有打印语句
        break

def main():
    # 测试数据加载和预处理
    processor = MNISTDataProcessor(batch_size=32)
    processed_train_loader, processed_test_loader = processor.get_processed_loaders()
    
    # 取并打印第一个批次的统计信息
    for data, targets in processed_train_loader:
        processor.print_batch_statistics(data, targets)
        break

# 在main函数中调用
if __name__ == "__main__":
    processor = MNISTDataProcessor(batch_size=32)
    verify_preprocessing(processor)