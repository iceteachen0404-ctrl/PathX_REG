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
import yaml
from typing import Dict, List, Tuple, Optional

# 导入CoCA框架
from coca_pytorch import CoCa

# 导入Prism模型
import sys
sys.path.append('./prism')
from modeling_prism import Prism
from configuring_prism import PrismConfig

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WSIDataset(Dataset):
    """WSI数据集类，用于加载WSI embeddings和对应的文本标签"""
    
    def __init__(self, 
                 embeddings_dir: str, 
                 labels_file: str, 
                 max_seq_len: int = 512,
                 image_dim: int = 2560):
        """
        Args:
            embeddings_dir: 包含.h5文件的目录路径
            labels_file: 包含文本标签的.json文件路径
            max_seq_len: 文本序列最大长度
            image_dim: 图像embedding维度
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.max_seq_len = max_seq_len
        self.image_dim = image_dim
        
        # 加载文本标签
        with open(labels_file, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        
        # 获取所有可用的embedding文件
        self.embedding_files = []
        for filename in self.labels.keys():
            h5_path = self.embeddings_dir / f"{filename}.h5"
            if h5_path.exists():
                self.embedding_files.append((filename, h5_path))
        
        logger.info(f"找到 {len(self.embedding_files)} 个有效的样本")
        
        # 创建简单的tokenizer（这里使用简单的字符级tokenizer，实际使用时可以替换为更复杂的tokenizer）
        self.vocab = self._build_vocab()
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
    def _build_vocab(self):
        """构建词汇表"""
        vocab = set()
        for text in self.labels.values():
            vocab.update(text.lower().split())
        # 添加特殊token
        special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
        vocab = special_tokens + sorted(list(vocab))
        return vocab
    
    def tokenize(self, text: str) -> List[int]:
        """简单的tokenization"""
        words = text.lower().split()
        tokens = [self.token_to_id.get(word, self.token_to_id['<unk>']) for word in words]
        return [self.token_to_id['<bos>']] + tokens + [self.token_to_id['<eos>']]
    
    def __len__(self):
        return len(self.embedding_files)
    
    def __getitem__(self, idx):
        filename, h5_path = self.embedding_files[idx]
        text = self.labels[filename]
        
        # 加载WSI embeddings
        with h5py.File(h5_path, 'r') as f:
            embeddings = f['embeddings'][:]  # 假设embeddings存储在'embeddings'键下
        
        # 确保embeddings是正确的形状
        if len(embeddings.shape) == 2:
            embeddings = embeddings.reshape(1, -1, self.image_dim)
        
        # Tokenize文本
        tokens = self.tokenize(text)
        tokens = tokens[:self.max_seq_len]
        
        # Padding
        if len(tokens) < self.max_seq_len:
            tokens.extend([self.token_to_id['<pad>']] * (self.max_seq_len - len(tokens)))
        
        return {
            'image_embeddings': torch.FloatTensor(embeddings),
            'text_tokens': torch.LongTensor(tokens),
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
        with torch.no_grad():  # 冻结Prism模型
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
    text_tokens = torch.stack([item['text_tokens'] for item in batch])
    
    # 其他信息
    texts = [item['text'] for item in batch]
    filenames = [item['filename'] for item in batch]
    
    return {
        'image_embeddings': image_embeddings,
        'text_tokens': text_tokens,
        'texts': texts,
        'filenames': filenames
    }

def train_epoch(model, dataloader, optimizer, scaler, device, epoch, config):
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
        
        if config['amp']['enabled']:
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
        else:
            loss = model(
                text=input_tokens,
                image_tokens=image_embeddings,
                labels=labels,
                return_loss=True
            )
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
    
    avg_loss = total_loss / num_batches
    logger.info(f'Epoch {epoch} - Average Loss: {avg_loss:.4f}')
    return avg_loss

def validate(model, dataloader, device, config):
    """验证模型"""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            image_embeddings = batch['image_embeddings'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            
            labels = text_tokens[:, 1:].contiguous()
            input_tokens = text_tokens[:, :-1].contiguous()
            
            if config['amp']['enabled']:
                with autocast():
                    loss = model(
                        text=input_tokens,
                        image_tokens=image_embeddings,
                        labels=labels,
                        return_loss=True
                    )
            else:
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

def load_config(config_path: str) -> Dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(config: Dict):
    """设置日志"""
    log_level = getattr(logging, config['logging']['level'])
    logging.basicConfig(level=log_level)
    
    if config['logging']['save_logs']:
        log_dir = Path(config['logging']['log_dir'])
        log_dir.mkdir(exist_ok=True)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_dir / 'training.log')
        file_handler.setLevel(log_level)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # 设置格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加到logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

def main():
    parser = argparse.ArgumentParser(description='使用配置文件微调多模态VLM模型')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    setup_logging(config)
    
    # 创建输出目录
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # 设置设备
    if config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f'cuda:{config["device"]["cuda_device"]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    logger.info(f'使用设备: {device}')
    
    # 加载Prism模型
    logger.info('加载Prism模型...')
    prism_config = PrismConfig.from_pretrained('./prism')
    prism_model = Prism.from_pretrained('./prism', config=prism_config)
    prism_model = prism_model.to(device)
    
    # 创建数据集
    logger.info('创建数据集...')
    dataset = WSIDataset(
        embeddings_dir=config['data']['embeddings_dir'],
        labels_file=config['data']['labels_file'],
        max_seq_len=config['data']['max_seq_len'],
        image_dim=config['data']['image_dim']
    )
    
    # 分割训练集和验证集
    dataset_size = len(dataset)
    val_size = int(dataset_size * config['data']['val_split'])
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=config['training']['num_workers']
    )
    
    # 创建CoCA模型
    logger.info('创建CoCA模型...')
    coca_model = CoCa(
        dim=config['model']['dim'],
        num_tokens=len(dataset.vocab),
        unimodal_depth=config['model']['unimodal_depth'],
        multimodal_depth=config['model']['multimodal_depth'],
        dim_latents=config['model']['dim_latents'],
        image_dim=config['model']['image_dim'],
        num_img_queries=config['model']['num_img_queries'],
        dim_head=config['model']['dim_head'],
        heads=config['model']['heads'],
        ff_mult=config['model']['ff_mult'],
        img_encoder=PrismImageEncoder(prism_model),
        caption_loss_weight=config['model']['caption_loss_weight'],
        contrastive_loss_weight=config['model']['contrastive_loss_weight'],
        pad_id=dataset.token_to_id['<pad>']
    )
    
    coca_model = coca_model.to(device)
    
    # 设置优化器
    if config['optimizer']['type'] == 'AdamW':
        optimizer = optim.AdamW(
            coca_model.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=tuple(config['optimizer']['betas']),
            eps=config['optimizer']['eps']
        )
    else:
        raise ValueError(f"不支持的优化器类型: {config['optimizer']['type']}")
    
    # 学习率调度器
    if config['scheduler']['type'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['training']['epochs']
        )
    else:
        raise ValueError(f"不支持的学习率调度器类型: {config['scheduler']['type']}")
    
    # 混合精度训练
    scaler = GradScaler() if config['amp']['enabled'] else None
    
    # 恢复训练
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume:
        logger.info(f'从检查点恢复训练: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        coca_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f'从epoch {start_epoch}开始恢复训练')
    
    # 训练循环
    logger.info('开始训练...')
    
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        # 训练
        train_loss = train_epoch(coca_model, train_loader, optimizer, scaler, device, epoch, config)
        
        # 验证
        val_loss = validate(coca_model, val_loader, device, config)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if config['checkpoint']['save_best'] and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': coca_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'vocab': dataset.vocab,
                'token_to_id': dataset.token_to_id,
                'config': config,
            }, output_dir / 'best_model.pth')
            logger.info(f'保存最佳模型，验证损失: {val_loss:.4f}')
        
        # 定期保存检查点
        if epoch % config['training']['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': coca_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'vocab': dataset.vocab,
                'token_to_id': dataset.token_to_id,
                'config': config,
            }, output_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # 保存最后一个epoch
        if config['checkpoint']['save_last'] and epoch == config['training']['epochs']:
            torch.save({
                'epoch': epoch,
                'model_state_dict': coca_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'vocab': dataset.vocab,
                'token_to_id': dataset.token_to_id,
                'config': config,
            }, output_dir / 'last_model.pth')
    
    logger.info('训练完成！')

if __name__ == '__main__':
    main() 