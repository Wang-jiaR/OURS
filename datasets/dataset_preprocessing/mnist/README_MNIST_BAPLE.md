# MNIST数据集在BAPLE框架中的集成说明

## 概述

本文檔說明如何將MNIST數據集集成到BAPLE框架中進行實驗。已創建了所有必要的文件，可以直接使用。

## 已創建的文件

### 1. 数据集类文件
**位置**: `datasets/mnist.py`
**功能**: 
- 继承`DatasetBase`类
- 实现数据加载逻辑
- 注册到`DATASET_REGISTRY`
- 支持few-shot学习
- 处理类别子采样

**关键方法**:
- `__init__()`: 初始化数据集
- `read_classnames()`: 读取类别名称
- `read_data()`: 读取图像数据
- `generate_fewshot_dataset()`: 生成few-shot数据集

### 2. 配置文件
**位置**: `configs/datasets/mnist.yaml`
**内容**:
```yaml
DATASET:
  NAME: "MNIST"
```

### 3. 类别名称文件
**位置**: `datasets/dataset_preprocessing/mnist/classnames.txt`
**内容**:
```
0 digit zero
1 digit one
2 digit two
3 digit three
4 digit four
5 digit five
6 digit six
7 digit seven
8 digit eight
9 digit nine
```

### 4. 预处理脚本
**位置**: `datasets/dataset_preprocessing/mnist/train_test_split_mnist.py`
**功能**:
- 解析MNIST文件名格式（子数据集名_编号_标签）
- 按7:3比例分割训练集和测试集
- 按数字类别(0-9)组织文件
- 调整图像大小为224x224像素
- 生成文本描述

## 数据集目录结构

运行预处理脚本后，将生成以下目录结构：
```
med-datasets/
└── mnist/
    ├── images/
    │   ├── train/
    │   │   ├── 0/          # 数字0的训练样本
    │   │   ├── 1/          # 数字1的训练样本
    │   │   ├── ...
    │   │   └── 9/          # 数字9的训练样本
    │   └── test/
    │       ├── 0/          # 数字0的测试样本
    │       ├── 1/          # 数字1的测试样本
    │       ├── ...
    │       └── 9/          # 数字9的测试样本
    ├── classnames.txt      # 类别名称文件
    ├── preprocessed.pkl    # 预处理缓存文件
    └── split_fewshot/      # few-shot数据缓存目录
```

## 使用步骤

### 步骤1: 准备数据
将MNIST图像文件按以下格式命名并放在mnist目录中：
```
子数据集名_编号_标签.扩展名
```
例如：
- `train_00001_5.png`
- `test_00123_0.jpg`
- `validation_00456_9.jpeg`

### 步骤2: 运行预处理脚本
```bash
cd med-datasets/mnist
python train_test_split_mnist.py
```

### 步骤3: 更新主程序导入
在`main.py`中添加：
```python
import datasets.mnist
```

### 步骤4: 更新评估脚本
在`scripts/eval.sh`中添加MNIST数据集支持：
```bash
elif [[ $DATASET == "mnist" ]] ; then
    TARGET_CLASSES=(0 1 2 3 4 5 6 7 8 9)
```

### 步骤5: 运行实验
```bash
# 训练
bash scripts/clip.sh mnist clip_ep50 16

# 评估
bash scripts/eval.sh clip mnist clip_ep50 16
```

## 数据集特点

### 1. 类别信息
- **类别数量**: 10个类别（数字0-9）
- **任务类型**: 手写数字识别
- **图像类型**: 灰度图像

### 2. 文件命名规范
- 支持多种图像格式：.png, .jpg, .jpeg
- 文件名格式：`子数据集名_编号_标签.扩展名`
- 标签范围：0-9

### 3. 预处理特性
- 图像大小调整：224x224像素
- 中心裁剪：非正方形图像进行中心裁剪
- 分层采样：确保每个类别在训练集和测试集中都有代表性
- 并行处理：使用多进程加速图像处理

### 4. 文本描述生成
为每个样本生成对应的文本描述：
- 格式：`A handwritten digit image of number [数字].`
- 示例：`A handwritten digit image of number 5.`

## 实验配置

### 训练配置
- **批次大小**: 16
- **学习率**: 0.02
- **训练轮数**: 50
- **优化器**: SGD
- **学习率调度**: Cosine

### Few-shot配置
- 支持1, 2, 4, 8, 16 shot实验
- 自动生成few-shot数据集
- 缓存预处理结果

### 评估配置
- 支持所有10个类别的目标攻击
- 后门攻击参数可配置
- 清洁和中毒样本分别评估

## 注意事项

1. **文件格式**: 确保图像文件格式正确且可读取
2. **命名规范**: 严格按照指定格式命名文件
3. **路径配置**: 正确设置数据集根目录路径
4. **依赖安装**: 确保所有必要的Python包已安装
5. **存储空间**: 预处理后的图像会占用更多存储空间

## 故障排除

### 常见问题
1. **文件名解析错误**: 检查文件名格式是否正确
2. **图像加载失败**: 确认图像文件完整且格式支持
3. **路径错误**: 检查数据集目录结构是否正确
4. **内存不足**: 减少并行处理的工作进程数

### 调试建议
1. 先运行测试脚本验证文件名解析
2. 检查生成的目录结构
3. 验证类别分布是否平衡
4. 确认图像大小调整是否正确

## 扩展功能

### 自定义配置
可以修改以下参数：
- 训练集比例：修改`train_ratio`参数
- 图像大小：修改`newsize`参数
- 文本描述：修改`prompt_engineering`函数
- 并行进程数：修改`num_workers`参数

### 添加新特性
- 数据增强：在预处理脚本中添加数据增强
- 类别平衡：实现更复杂的采样策略
- 多模态：添加其他模态的数据支持 