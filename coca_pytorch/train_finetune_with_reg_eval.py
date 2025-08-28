import os
import json
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import pdb

# 导入CoCA框架
from coca_pytorch import CoCa

# 导入Prism模型
import sys
sys.path.append('./prism')
from prism.modeling_prism import Prism
from prism.configuring_prism import PrismConfig

# 导入BioGPT tokenizer
from transformers import BioGptTokenizer

# 导入REG_evaluator
from metric.eval import REG_Evaluator

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WSIDataset(Dataset):
    """WSI数据集类，用于加载WSI embeddings和对应的文本标签"""
    
    def __init__(self, 
                 embeddings_dir: str, 
                 labels_file: str, 
                 image_dim: int = 2560,
                 max_seq_len: int = 512):
        """
        Args:
            embeddings_dir: 包含.h5文件的目录路径
            labels_file: 包含文本标签的.json文件路径
            image_dim: 图像embedding维度
            max_seq_len: 文本序列最大长度
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.image_dim = image_dim
        self.max_seq_len = max_seq_len
        
        # 加载文本标签
        with open(labels_file, 'r', encoding='utf-8') as f:
            reg_label = json.load(f)
            self.labels = {}
            for item in reg_label:
                id = item['id']
                report = item['report']
                self.labels[id] = report
        
        # 获取所有可用的embedding文件
        self.embedding_files = []
        for filename in self.labels.keys():
            h5_path = self.embeddings_dir / f"{filename.split('.')[0]}.h5"
            if h5_path.exists():
                self.embedding_files.append((filename, h5_path))
        
        logger.info(f"找到 {len(self.embedding_files)} 个有效的样本")
        # 初始化tokenizer
        commit_hash = 'eb0d815e95434dc9e3b78f464e52b899bee7d923'
        self.tokenizer = BioGptTokenizer.from_pretrained('/data/case_level/open_source_model/model/biogpt', revision=commit_hash)
        self.pad_id = self.tokenizer.pad_token_id
        self.bos_id = self.tokenizer.eos_token_id  # BioGPT使用EOS作为BOS
        self.eos_id = self.tokenizer.eos_token_id
        self.cls_id = 42383  # BioGPT的CLS token ID
    
    def tokenize_text(self, text: str) -> torch.Tensor:
        """将文本转换为token IDs"""
        # 添加EOS token
        text_with_eos = text + self.tokenizer.eos_token
        
        # Tokenize
        tokenized = self.tokenizer(
            text=text_with_eos,
            add_special_tokens=True,
            padding=False,
            return_tensors='pt',
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        
        token_ids = tokenized['input_ids'][0]
        
        # 截断到最大长度
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        
        return token_ids
    
    def __len__(self):
        return len(self.embedding_files)
    
    def __getitem__(self, idx):
        filename, h5_path = self.embedding_files[idx]
        text = self.labels[filename]
        
        # 加载WSI embeddings
        with h5py.File(h5_path, 'r') as f:
            embeddings = f['features'][:]  # 假设embeddings存储在'features'键下
        
        # 确保embeddings是正确的形状
        if len(embeddings.shape) == 2:
            embeddings = embeddings.reshape(1, -1, self.image_dim)
        
        # Tokenize文本
        text_tokens = self.tokenize_text(text)
        
        return {
            'image_embeddings': torch.FloatTensor(embeddings),
            'text_tokens': text_tokens,
            'text': text,
            'filename': filename
        }

class PrismImageEncoder(nn.Module):
    """Prism图像编码器包装器，用于CoCA框架"""
    
    def __init__(self, prism_model: Prism):
        super().__init__()
        self.prism_model = prism_model
        
    def forward(self, tile_embeddings):
        """前向传播，返回图像表示"""
        reprs = self.prism_model.slide_representations(tile_embeddings)
        return reprs['image_embedding']

def collate_fn(batch):
    """数据批处理函数"""
    # 获取最大序列长度
    max_seq_len = max(item['image_embeddings'].shape[1] for item in batch)
    
    # 处理图像embeddings
    image_embeddings = []
    for item in batch:
        # 填充图像embeddings到最大长度
        embeddings = item['image_embeddings']
        if embeddings.shape[1] < max_seq_len:
            padding = torch.zeros(1, max_seq_len - embeddings.shape[1], embeddings.shape[2])
            embeddings = torch.cat([embeddings, padding], dim=1)
        image_embeddings.append(embeddings)
    
    image_embeddings = torch.cat(image_embeddings, dim=0)
    
    # 处理文本tokens
    text_tokens = [item['text_tokens'] for item in batch]
    
    # 获取最大文本长度
    max_text_len = max(len(tokens) for tokens in text_tokens)
    
    # Padding文本tokens
    padded_text_tokens = []
    for tokens in text_tokens:
        if len(tokens) < max_text_len:
            # 使用pad_id进行padding
            padding = torch.full((max_text_len - len(tokens),), 1, dtype=tokens.dtype)  # pad_id = 1
            tokens = torch.cat([tokens, padding])
        padded_text_tokens.append(tokens)
    
    text_tokens = torch.stack(padded_text_tokens)
    
    # 其他信息
    texts = [item['text'] for item in batch]
    filenames = [item['filename'] for item in batch]
    
    return {
        'image_embeddings': image_embeddings,
        'text_tokens': text_tokens,
        'texts': texts,
        'filenames': filenames
    }

def generate_text(model, image_embeddings, tokenizer, max_length=100, device='cuda'):
    """使用模型生成文本"""
    model.eval()
    with torch.no_grad():
        # 生成文本
        generated_tokens = model.module.generate(
            image_tokens=image_embeddings,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # 解码生成的文本
        generated_texts = []
        for tokens in generated_tokens:
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            generated_texts.append(text)
    
    return generated_texts

def evaluate_with_reg_evaluator(model, dataloader, device, reg_evaluator, tokenizer, num_samples=50):
    """使用REG_evaluator评估模型生成的文本质量"""
    model.eval()
    eval_pairs = []
    
    # 随机选择一些样本进行评估
    all_samples = list(dataloader)
    if len(all_samples) > num_samples:
        import random
        selected_samples = random.sample(all_samples, num_samples)
    else:
        selected_samples = all_samples
    
    logger.info(f"使用REG_evaluator评估 {len(selected_samples)} 个样本...")
    
    with torch.no_grad():
        for batch in tqdm(selected_samples, desc='REG Evaluation'):
            image_embeddings = batch['image_embeddings'].to(device)
            reference_texts = batch['texts']
            
            # 生成文本
            generated_texts = generate_text(model, image_embeddings, tokenizer, device=device)
            
            # 创建评估对
            for ref_text, hyp_text in zip(reference_texts, generated_texts):
                if hyp_text.strip():  # 确保生成的文本不为空
                    eval_pairs.append((ref_text, hyp_text))
    
    # 计算REG评分
    if eval_pairs:
        reg_score = reg_evaluator.evaluate_dummy(eval_pairs)
        logger.info(f'REG评分: {reg_score:.4f}')
        return reg_score
    else:
        logger.warning("没有有效的评估对")
        return 0.0

def train_epoch(model, dataloader, optimizer, scaler, device, epoch, reg_evaluator=None, tokenizer=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        image_embeddings = batch['image_embeddings'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        
        # 创建标签（用于caption loss）
        labels = text_tokens[:, 1:].contiguous()  # 去掉BOS token
        input_tokens = text_tokens[:, :-1].contiguous()  # 去掉EOS token
        
        optimizer.zero_grad()
        
        with autocast():
            loss = model(
                text=input_tokens,
                image_tokens=image_embeddings,
                labels=labels,
                return_loss=True
            )
        loss = loss.mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
    
    avg_loss = total_loss / num_batches
    logger.info(f'Epoch {epoch} - Average Loss: {avg_loss:.4f}')
    return avg_loss

def validate(model, dataloader, device, reg_evaluator=None, tokenizer=None):
    """验证模型"""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            image_embeddings = batch['image_embeddings'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            
            # 创建标签（用于caption loss）
            labels = text_tokens[:, 1:].contiguous()  # 去掉BOS token
            input_tokens = text_tokens[:, :-1].contiguous()  # 去掉EOS token
            
            with autocast():
                loss = model(
                    text=input_tokens,
                    image_tokens=image_embeddings,
                    labels=labels,
                    return_loss=True
                )
            loss = loss.mean()
            
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    logger.info(f'Validation Loss: {avg_loss:.4f}')
    
    # 如果提供了REG_evaluator，计算REG评分
    reg_score = 0.0
    if reg_evaluator is not None and tokenizer is not None:
        reg_score = evaluate_with_reg_evaluator(model, dataloader, device, reg_evaluator, tokenizer)
    
    return avg_loss, reg_score

def main():
    parser = argparse.ArgumentParser(description='微调多模态VLM模型（集成REG_evaluator）')
    parser.add_argument('--embeddings_dir', type=str, default='/data/slide_files/nas/vol2/Public_Data/reg2025/REG_train/normal/trident_prcessed/20x_256px_0px_overlap/features_virchow/', help='WSI embeddings目录')
    parser.add_argument('--labels_file', type=str, default='/data/slide_files/nas/vol2/Public_Data/reg2025/REG_train/train_reg2025.json', help='文本标签JSON文件')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--max_seq_len', type=int, default=512, help='最大序列长度')
    parser.add_argument('--val_split', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--save_every', type=int, default=5, help='每N个epoch保存一次')
    parser.add_argument('--reg_eval_freq', type=int, default=2, help='每N个epoch进行一次REG评估')
    parser.add_argument('--embedding_model', type=str, default='/data/case_level/open_source_model/model/Llama3-OpenBioLLM-8B', help='REG评估器使用的embedding模型')
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='使用的GPU编号，例如: "0", "0,1", "1,2,3"')
    args = parser.parse_args()
    import os

    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 初始化REG_evaluator
    logger.info('初始化REG_evaluator...')
    reg_evaluator = REG_Evaluator(embedding_model=args.embedding_model)
    
    # 加载Prism模型
    logger.info('加载Prism模型...')
    prism_config = PrismConfig.from_pretrained('./prism')
    prism_model = Prism.from_pretrained('./prism', config=prism_config)
    prism_model = prism_model.to(device)
    
    # 创建数据集
    logger.info('创建数据集...')
    dataset = WSIDataset(
        embeddings_dir=args.embeddings_dir,
        labels_file=args.labels_file,
        image_dim=2560,  # Prism的tile embedding维度
        max_seq_len=args.max_seq_len
    )
    # pdb.set_trace()  # 进入调试模式

    # 分割训练集和验证集
    dataset_size = len(dataset)
    val_size = int(dataset_size * args.val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # 创建CoCA模型
    logger.info('创建CoCA模型...')
    coca_model = CoCa(
        dim=1024,  # 与BioGPT的hidden_size一致
        num_tokens=42384,  # BioGPT的词汇表大小
        unimodal_depth=6,
        multimodal_depth=6,
        dim_latents=5120,  # 与Prism配置一致
        image_dim=2560,  # Prism的image_embedding维度
        num_img_queries=256,
        dim_head=64,
        heads=16,
        ff_mult=4,
        img_encoder=PrismImageEncoder(prism_model),
        caption_loss_weight=1.0,
        contrastive_loss_weight=1.0,
        pad_id=1  # BioGPT的pad_id
    )
    
    coca_model = coca_model.to(device)

    # 使用多GPU
    if torch.cuda.device_count() > 1:
        logger.info(f'使用 {torch.cuda.device_count()} 个GPU进行DataParallel')
        coca_model = nn.DataParallel(coca_model)
    
    # 设置优化器
    optimizer = optim.AdamW(coca_model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 训练循环
    logger.info('开始训练...')
    best_val_loss = float('inf')
    best_reg_score = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_epoch(coca_model, train_loader, optimizer, scaler, device, epoch)
        
        # 验证
        val_loss, reg_score = validate(coca_model, val_loader, device, reg_evaluator, train_dataset.dataset.tokenizer)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型（基于验证损失）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': coca_model.state_dict() if isinstance(coca_model, nn.DataParallel) else coca_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'reg_score': reg_score,
                'tokenizer_config': {
                    'name': 'biogpt',
                    'vocab_size': train_dataset.tokenizer.vocab_size,
                    'pad_id': train_dataset.pad_id,
                    'bos_id': train_dataset.bos_id,
                    'eos_id': train_dataset.eos_id,
                    'cls_id': train_dataset.cls_id
                },
            }, output_dir / 'best_model_loss.pth')
            logger.info(f'保存最佳模型（基于损失），验证损失: {val_loss:.4f}, REG评分: {reg_score:.4f}')
        
        # 保存最佳模型（基于REG评分）
        if reg_score > best_reg_score:
            best_reg_score = reg_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': coca_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'reg_score': reg_score,
                'tokenizer_config': {
                    'name': 'biogpt',
                    'vocab_size': train_dataset.tokenizer.vocab_size,
                    'pad_id': train_dataset.pad_id,
                    'bos_id': train_dataset.bos_id,
                    'eos_id': train_dataset.eos_id,
                    'cls_id': train_dataset.cls_id
                },
            }, output_dir / 'best_model_reg.pth')
            logger.info(f'保存最佳模型（基于REG评分），验证损失: {val_loss:.4f}, REG评分: {reg_score:.4f}')
        
        # 定期保存检查点
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': coca_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'reg_score': reg_score,
                'tokenizer_config': {
                    'name': 'biogpt',
                    'vocab_size': train_dataset.tokenizer.vocab_size,
                    'pad_id': train_dataset.pad_id,
                    'bos_id': train_dataset.bos_id,
                    'eos_id': train_dataset.eos_id,
                    'cls_id': train_dataset.cls_id
                },
            }, output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    logger.info('训练完成！')
    logger.info(f'最佳验证损失: {best_val_loss:.4f}')
    logger.info(f'最佳REG评分: {best_reg_score:.4f}')

if __name__ == '__main__':
    main() 