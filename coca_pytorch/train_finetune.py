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
from transformers import AutoModel
# 导入CoCA框架
from coca_pytorch import CoCa

# 导入Prism模型
import sys
sys.path.append('./prism')
from prism.modeling_prism import Prism
from prism.configuring_prism import PrismConfig

# 导入BioGPT tokenizer
from transformers import BioGptTokenizer

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
        self.embeddings_dir = embeddings_dir
        self.image_dim = image_dim
        self.max_seq_len = max_seq_len
        
        # 加载文本标签
        with open(labels_file, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        
        # 获取所有可用的embedding文件
        self.embedding_files = []
        for dis in os.listdir(self.embeddings_dir):
            for file in os.listdir(self.embeddings_dir+'/'+dis):
                h5_path = self.embeddings_dir + '/' + dis + '/' + file
                if os.path.exists(h5_path):
                    self.embedding_files.append((file, h5_path))
        
        logger.info(f"找到 {len(self.embedding_files)} 个有效的样本")
        
        # 初始化tokenizer
        commit_hash = 'eb0d815e95434dc9e3b78f464e52b899bee7d923'
        self.tokenizer = BioGptTokenizer.from_pretrained('microsoft/biogpt', revision=commit_hash)
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
        emb = item['image_embeddings']
        if emb.shape[1] < max_seq_len:
            # Padding
            pad_size = max_seq_len - emb.shape[1]
            emb = torch.cat([emb, torch.zeros(emb.shape[0], pad_size, emb.shape[2])], dim=1)
        image_embeddings.append(emb)
    
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

def train_epoch(model, dataloader, optimizer, scaler, device, epoch):
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

def validate(model, dataloader, device):
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
            
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    logger.info(f'Validation Loss: {avg_loss:.4f}')
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description='微调多模态VLM模型')
    parser.add_argument('--embeddings_dir', type=str, default='D:/BRACS/virchow/train', help='WSI embeddings目录')
    parser.add_argument('--labels_file', type=str, default='D:/BRACS/conchv15/all_text_label.json', help='文本标签JSON文件')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--max_seq_len', type=int, default=512, help='最大序列长度')
    parser.add_argument('--val_split', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--save_every', type=int, default=5, help='每N个epoch保存一次')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 加载Prism模型 
    logger.info('加载Prism模型...')
    # prism_config = PrismConfig.from_pretrained('./prism')
    #prism_model = Prism.from_pretrained('./prism', config=prism_config)
    prism_model = AutoModel.from_pretrained('paige-ai/Prism', trust_remote_code=True)
    prism_model = prism_model.to(device)
    
    # 创建数据集
    logger.info('创建数据集...')
    train_dataset = WSIDataset(
        embeddings_dir='D:/BRACS/virchow/train',
        labels_file=args.labels_file,
        image_dim=2560,  # Prism的tile embedding维度
        max_seq_len=args.max_seq_len
    )
    
    val_dataset = WSIDataset(
        embeddings_dir='D:/BRACS/virchow/val',
        labels_file=args.labels_file,
        image_dim=2560,  # Prism的tile embedding维度
        max_seq_len=args.max_seq_len
    )
    
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=1
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=1
    )
    
    # 创建CoCA模型
    logger.info('创建CoCA模型...')
    coca_model = CoCa(
        dim=1024,  # 与BioGPT的hidden_size一致
        num_tokens=42384,  # BioGPT的词汇表大小
        unimodal_depth=6,
        multimodal_depth=6,
        dim_latents=5120,  # 与Prism配置一致
        image_dim=1280,  # Prism的image_embedding维度
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
    
    # 设置优化器
    optimizer = optim.AdamW(coca_model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 训练循环
    logger.info('开始训练...')
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_epoch(coca_model, train_loader, optimizer, scaler, device, epoch)
        
        # 验证
        val_loss = validate(coca_model, val_loader, device)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': coca_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'tokenizer_config': {
                    'name': 'biogpt',
                    'vocab_size': train_dataset.tokenizer.vocab_size,
                    'pad_id': train_dataset.pad_id,
                    'bos_id': train_dataset.bos_id,
                    'eos_id': train_dataset.eos_id,
                    'cls_id': train_dataset.cls_id
                },
            }, output_dir / 'best_model.pth')
            logger.info(f'保存最佳模型，验证损失: {val_loss:.4f}')
        
        # 定期保存检查点
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': coca_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
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

if __name__ == '__main__':
    main() 