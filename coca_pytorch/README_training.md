# 多模态VLM模型微调指南

本项目提供了用于微调多模态视觉语言模型（VLM）的完整训练框架，集成了CoCA框架和Prism病理模型。

## 项目结构

```
.
├── coca_pytorch/           # CoCA框架
├── prism/                  # Prism病理模型
├── train_finetune.py       # 基础训练脚本
├── train_with_config.py    # 支持YAML配置的训练脚本
├── inference.py            # 推理脚本
├── config.yaml             # 训练配置文件
└── README_training.md      # 本文件
```

## 环境要求

```bash
pip install torch torchvision
pip install transformers
pip install h5py
pip install tqdm
pip install pyyaml
pip install einops
```

## 数据准备

### 1. WSI Embeddings (.h5文件)

将WSI的tile embeddings保存为.h5格式，每个文件包含：
- `embeddings`: 形状为 `(num_tiles, 2560)` 的numpy数组
- 文件名格式：`{image_id}.h5`

### 2. 文本标签 (.json文件)

创建包含图像-文本对应关系的JSON文件：
```json
{
    "image_001": "浸润性导管癌，中等分化",
    "image_002": "良性乳腺组织",
    "image_003": "小叶原位癌"
}
```

## 使用方法

### 1. 基础训练（命令行参数）

```bash
python train_finetune.py \
    --embeddings_dir ./data/embeddings \
    --labels_file ./data/labels.json \
    --output_dir ./checkpoints \
    --batch_size 4 \
    --epochs 10 \
    --lr 1e-4
```

### 2. 使用配置文件训练（推荐）

首先修改 `config.yaml` 文件中的参数，然后运行：

```bash
# 开始新训练
python train_with_config.py --config config.yaml

# 从检查点恢复训练
python train_with_config.py --config config.yaml --resume ./checkpoints/checkpoint_epoch_5.pth
```

### 3. 推理

```bash
python inference.py \
    --checkpoint ./checkpoints/best_model.pth \
    --embeddings_file ./data/embeddings/image_001.h5 \
    --text "浸润性导管癌" \
    --max_length 100
```

## 配置文件说明

### 数据配置
```yaml
data:
  embeddings_dir: "./data/embeddings"  # WSI embeddings目录
  labels_file: "./data/labels.json"    # 文本标签JSON文件
  max_seq_len: 512                     # 文本序列最大长度
  image_dim: 2560                      # 图像embedding维度
  val_split: 0.1                       # 验证集比例
```

### 模型配置
```yaml
model:
  dim: 1024                            # 模型维度
  unimodal_depth: 6                    # 单模态层深度
  multimodal_depth: 6                  # 多模态层深度
  dim_latents: 5120                    # 潜在空间维度
  image_dim: 1280                      # 图像embedding维度
  caption_loss_weight: 1.0             # 描述损失权重
  contrastive_loss_weight: 1.0         # 对比损失权重
```

### 训练配置
```yaml
training:
  batch_size: 4                        # 批次大小
  epochs: 10                           # 训练轮数
  learning_rate: 1e-4                  # 学习率
  weight_decay: 0.01                   # 权重衰减
  save_every: 5                        # 每N个epoch保存一次
```

## 模型架构

### CoCA框架集成
- 使用CoCA框架作为主要的训练架构
- 集成Prism模型作为图像编码器
- 支持对比学习和描述生成任务

### Prism图像编码器
- 冻结Prism模型的权重
- 使用Prism的slide_representations方法
- 输出1280维的图像表示

### 损失函数
1. **Caption Loss**: 交叉熵损失，用于文本生成
2. **Contrastive Loss**: 对比损失，用于图像-文本对齐

## 训练监控

### 日志
- 训练日志保存在 `./logs/training.log`
- 控制台实时显示训练进度
- 包含损失值、学习率等信息

### 检查点
- `best_model.pth`: 验证损失最低的模型
- `checkpoint_epoch_N.pth`: 定期保存的检查点
- `last_model.pth`: 最后一个epoch的模型

## 性能优化

### 混合精度训练
- 默认启用FP16混合精度训练
- 可显著减少显存使用和训练时间

### 数据加载优化
- 多进程数据加载
- 动态批处理大小调整
- 内存映射文件读取

## 故障排除

### 常见问题

1. **显存不足**
   - 减少batch_size
   - 启用混合精度训练
   - 使用梯度累积

2. **数据加载错误**
   - 检查.h5文件格式
   - 确认JSON文件编码为UTF-8
   - 验证文件路径正确性

3. **模型收敛慢**
   - 调整学习率
   - 检查数据质量
   - 增加训练轮数

### 调试模式

```bash
# 启用详细日志
export PYTHONPATH=.
python train_with_config.py --config config.yaml 2>&1 | tee debug.log
```

## 扩展功能

### 自定义Tokenizer
可以替换简单的词汇表构建方法，使用更复杂的tokenizer：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
```

### 自定义损失函数
可以在训练脚本中添加自定义损失函数：

```python
def custom_loss(pred, target):
    # 自定义损失计算
    return loss
```

### 多GPU训练
支持分布式训练，需要修改训练脚本以支持多GPU：

```python
model = torch.nn.DataParallel(model)
```

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进这个训练框架。 