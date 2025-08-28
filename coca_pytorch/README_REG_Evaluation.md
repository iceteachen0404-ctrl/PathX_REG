# REG评估器集成训练指南

本指南介绍如何将REG_evaluator集成到CoCA模型的微调过程中，用于评估生成文本的质量。

## 概述

REG_evaluator是一个综合的文本质量评估器，它结合了以下四个指标来计算ranking score：

1. **嵌入相似度评分** (30%权重) - 使用预训练模型计算参考文本和生成文本的语义相似度
2. **关键词匹配评分** (40%权重) - 使用spaCy提取医学关键词，计算Jaccard相似度
3. **BLEU-4评分** (7.5%权重) - 计算4-gram精确匹配
4. **ROUGE评分** (7.5%权重) - 计算最长公共子序列

最终的ranking score计算公式：
```
ranking_score = 0.15 * (rouge_score + bleu_score) + 0.4 * keyword_score + 0.3 * embedding_score
```

## 重要更新

### 文本Tokenization修复
**最新版本修复了文本处理问题**：
- 文本现在正确转换为token IDs，而不是直接传入字符串
- 使用BioGPT tokenizer进行正确的tokenization
- 支持文本序列长度限制和padding
- 正确处理BOS/EOS token和标签创建

## 文件说明

### 1. 核心评估器
- `metric/eval.py` - REG_evaluator的实现，包含EmbeddingEvaluator、KeywordEvaluator和REG_Evaluator类

### 2. 训练脚本
- `train_finetune.py` - 基础版本的训练脚本（已修复tokenization）
- `train_finetune_with_reg_eval.py` - 集成REG评估器的训练脚本（已修复tokenization）
- `train_with_reg_config.py` - 使用配置文件的训练脚本，更灵活

### 3. 配置文件
- `config_reg_eval.yaml` - REG评估器的配置文件

### 4. 测试脚本
- `test_tokenization.py` - 测试tokenization功能
- `test_reg_evaluator.py` - 测试REG评估器功能

## 使用方法

### 测试Tokenization功能
```bash
python test_tokenization.py
```

### 方法1：使用基础训练脚本

```bash
python train_finetune.py \
    --embeddings_dir /path/to/embeddings \
    --labels_file /path/to/labels.json \
    --output_dir ./checkpoints \
    --batch_size 4 \
    --epochs 10 \
    --lr 1e-4 \
    --max_seq_len 512
```

### 方法2：使用集成REG评估器的训练脚本

```bash
python train_finetune_with_reg_eval.py \
    --embeddings_dir /path/to/embeddings \
    --labels_file /path/to/labels.json \
    --output_dir ./checkpoints \
    --batch_size 4 \
    --epochs 10 \
    --lr 1e-4 \
    --embedding_model dmis-lab/biobert-v1.1 \
    --max_seq_len 512
```

### 方法3：使用配置文件训练脚本

```bash
python train_with_reg_config.py \
    --embeddings_dir /path/to/embeddings \
    --labels_file /path/to/labels.json \
    --output_dir ./checkpoints \
    --config config_reg_eval.yaml
```

## 配置选项

### REG评估器配置

在`config_reg_eval.yaml`中可以配置以下参数：

```yaml
reg_evaluator:
  # 嵌入模型选择
  embedding_model: 'dmis-lab/biobert-v1.1'  # 可选模型见下方
  
  # Spacy模型
  spacy_model: 'en_core_sci_lg'
  
  # 评估频率
  eval_frequency: 2  # 每N个epoch进行一次REG评估
  
  # 评估样本数量
  num_eval_samples: 50  # 用于评估的样本数量
  
  # 文本生成参数
  generation:
    max_length: 100
    temperature: 0.7
    top_p: 0.9
    do_sample: true
```

### 支持的嵌入模型

REG_evaluator支持以下预训练模型：

1. `dmis-lab/biobert-v1.1` - BioBERT模型，适合生物医学文本
2. `aaditya/Llama3-OpenBioLLM-8B` - 生物医学LLaMA模型
3. `openai-communitya/gpt2` - GPT-2模型
4. `meta-llama/Meta-Llama-3.1-8B-Instruct` - LLaMA-3指令微调模型
5. `NeuML/pubmedbert-base-embeddings` - PubMedBERT嵌入模型

## 训练过程

### 1. 文本处理流程

1. **Tokenization**: 使用BioGPT tokenizer将文本转换为token IDs
2. **添加特殊token**: 自动添加EOS token
3. **长度限制**: 截断到最大序列长度
4. **Batch处理**: 在collate函数中进行padding
5. **标签创建**: 正确创建输入和标签序列

### 2. 模型保存策略

训练过程中会保存两种最佳模型：

- **基于验证损失的最佳模型** (`best_model_loss.pth`) - 当验证损失降低时保存
- **基于REG评分的最佳模型** (`best_model_reg.pth`) - 当REG评分提高时保存

### 3. 评估流程

每个epoch的验证阶段包括：

1. 计算验证损失
2. 使用模型生成文本
3. 使用REG_evaluator评估生成文本质量
4. 记录REG评分

### 4. 日志输出

训练过程中会输出以下信息：

```
Epoch 1 - Average Loss: 2.3456
Validation Loss: 2.1234
使用REG_evaluator评估 50 个样本...
REG评分: 0.6789
保存最佳模型（基于损失），验证损失: 2.1234, REG评分: 0.6789
保存最佳模型（基于REG评分），验证损失: 2.1234, REG评分: 0.6789
```

## 依赖安装

确保安装以下依赖：

```bash
pip install transformers torch scikit-learn tqdm spacy scispacy pyyaml
python -m spacy download en_core_sci_lg
```

## 注意事项

1. **内存使用**：REG评估需要加载额外的预训练模型，会增加内存使用量
2. **评估时间**：REG评估会增加验证时间，特别是当评估样本数量较大时
3. **模型选择**：建议根据您的数据领域选择合适的嵌入模型
4. **Spacy模型**：确保安装了正确的spaCy模型，医学文本推荐使用`en_core_sci_lg`
5. **Tokenization**：确保使用正确的BioGPT tokenizer版本和commit hash

## 自定义评估

如果需要自定义评估权重或添加新的评估指标，可以修改`metric/eval.py`中的`REG_Evaluator.evaluate_text`方法：

```python
def evaluate_text(self, ref_text, hyp_text):
    emb_score   = self.embedding_eval.get_score(ref_text,hyp_text)
    key_score   = self.key_eval.get_score(ref_text,hyp_text)
    bleu_score  = self.get_bleu4(ref_text,hyp_text)
    rouge_score = self.get_rouge(ref_text,hyp_text)

    # 自定义权重
    ranking_score = 0.15*(rouge_score+bleu_score) + 0.4*key_score + 0.3*emb_score
    return ranking_score
```

## 故障排除

### 常见问题

1. **Spacy模型未找到**：
   ```bash
   python -m spacy download en_core_sci_lg
   ```

2. **内存不足**：
   - 减少`num_eval_samples`
   - 使用更小的嵌入模型
   - 减少batch_size

3. **评估速度慢**：
   - 减少`num_eval_samples`
   - 增加`eval_frequency`
   - 使用更快的嵌入模型

4. **生成文本为空**：
   - 检查tokenizer配置
   - 调整生成参数（temperature, top_p）
   - 增加max_length

5. **Tokenization错误**：
   - 确保使用正确的BioGPT tokenizer版本
   - 检查commit hash是否正确
   - 运行`test_tokenization.py`验证功能 