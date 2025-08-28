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
from prism.modeling_prism import Prism
from prism.configuring_prism import PrismConfig

# 导入BioGPT tokenizer
from transformers import BioGptTokenizer

# 导入REG_evaluator
from metric.eval import REG_Evaluator

def setup_logging(config: Dict):
    """设置日志"""
    log_level = getattr(logging, config['logging']['level'])
    logging.basicConfig(level=log_level)
    return logging.getLogger(__name__)

class WSIDataset(Dataset):
    """WSI数据集类，用于加载WSI embeddings和对应的文本标签"""
    
    def __init__(self, 
                 embeddings_dir: str, 
                 labels_file: str, 
                 config: Dict):
        """
        Args:
            embeddings_dir: 包含.h5文件的目录路径
            labels_file: 包含文本标签的.json文件路径
            config: 配置字典
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.image_dim = config['data']['image_dim']
        
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
        
        # 初始化tokenizer
        self.tokenizer = BioGptTokenizer.from_pretrained('microsoft/biogpt')
        self.pad_id = self.tokenizer.pad_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.cls_id = self.tokenizer.cls_token_id
    
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
        
        return {
            'image_embeddings': torch.FloatTensor(embeddings),
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
    texts = []
    filenames = []
    
    for item in batch:
        # 填充图像embeddings到最大长度
        embeddings = item['image_embeddings']
        if embeddings.shape[1] < max_seq_len:
            padding = torch.zeros(1, max_seq_len - embeddings.shape[1], embeddings.shape[2])
            embeddings = torch.cat([embeddings, padding], dim=1)
        image_embeddings.append(embeddings)
        texts.append(item['text'])
        filenames.append(item['filename'])
    
    return {
        'image_embeddings': torch.cat(image_embeddings, dim=0),
        'texts': texts,
        'filenames': filenames
    }

def generate_text(model, image_embeddings, tokenizer, config, device='cuda'):
    """使用模型生成文本"""
    model.eval()
    gen_config = config['reg_evaluator']['generation']
    
    with torch.no_grad():
        # 生成文本
        generated_tokens = model.generate(
            image_tokens=image_embeddings,
            max_length=gen_config['max_length'],
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=gen_config['do_sample'],
            temperature=gen_config['temperature'],
            top_p=gen_config['top_p']
        )
        
        # 解码生成的文本
        generated_texts = []
        for tokens in generated_tokens:
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            generated_texts.append(text)
    
    return generated_texts

def evaluate_with_reg_evaluator(model, dataloader, device, reg_evaluator, tokenizer, config):
    """使用REG_evaluator评估模型生成的文本质量"""
    model.eval()
    eval_pairs = []
    num_samples = config['reg_evaluator']['num_eval_samples']
    
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
            generated_texts = generate_text(model, image_embeddings, tokenizer, config, device=device)
            
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

def train_epoch(model, dataloader, optimizer, scaler, device, epoch, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        image_embeddings = batch['image_embeddings'].to(device)
        texts = batch['texts']
        
        optimizer.zero_grad()
        
        if config['training']['amp']['enabled']:
            with autocast():
                loss = model(
                    text=texts,
                    image_tokens=image_embeddings,
                    return_loss=True
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(
                text=texts,
                image_tokens=image_embeddings,
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

def validate(model, dataloader, device, config, reg_evaluator=None, tokenizer=None):
    """验证模型"""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            image_embeddings = batch['image_embeddings'].to(device)
            texts = batch['texts']
            
            if config['training']['amp']['enabled']:
                with autocast():
                    loss = model(
                        text=texts,
                        image_tokens=image_embeddings,
                        return_loss=True
                    )
            else:
                loss = model(
                    text=texts,
                    image_tokens=image_embeddings,
                    return_loss=True
                )
            
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    logger.info(f'Validation Loss: {avg_loss:.4f}')
    
    # 如果提供了REG_evaluator，计算REG评分
    reg_score = 0.0
    if reg_evaluator is not None and tokenizer is not None:
        reg_score = evaluate_with_reg_evaluator(model, dataloader, device, reg_evaluator, tokenizer, config)
    
    return avg_loss, reg_score

def load_config(config_path: str) -> Dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='微调多模态VLM模型（集成REG_evaluator，使用配置文件）')
    parser.add_argument('--embeddings_dir', type=str, required=True, help='WSI embeddings目录')
    parser.add_argument('--labels_file', type=str, required=True, help='文本标签JSON文件')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='输出目录')
    parser.add_argument('--config', type=str, default='config_reg_eval.yaml', help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    global logger
    logger = setup_logging(config)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 初始化REG_evaluator
    logger.info('初始化REG_evaluator...')
    reg_config = config['reg_evaluator']
    reg_evaluator = REG_Evaluator(
        embedding_model=reg_config['embedding_model'],
        spacy_model=reg_config['spacy_model']
    )
    
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
        config=config
    )
    
    # 分割训练集和验证集
    dataset_size = len(dataset)
    val_size = int(dataset_size * config['training']['val_split'])
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
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # 创建CoCA模型
    logger.info('创建CoCA模型...')
    model_config = config['model']
    coca_model = CoCa(
        dim=model_config['dim'],
        num_tokens=model_config['num_tokens'],
        unimodal_depth=model_config['unimodal_depth'],
        multimodal_depth=model_config['multimodal_depth'],
        dim_latents=model_config['dim_latents'],
        image_dim=model_config['image_dim'],
        num_img_queries=model_config['num_img_queries'],
        dim_head=model_config['dim_head'],
        heads=model_config['heads'],
        ff_mult=model_config['ff_mult'],
        img_encoder=PrismImageEncoder(prism_model),
        caption_loss_weight=model_config['caption_loss_weight'],
        contrastive_loss_weight=model_config['contrastive_loss_weight'],
        pad_id=model_config['pad_id']
    )
    
    coca_model = coca_model.to(device)
    
    # 设置优化器
    optimizer = optim.AdamW(
        coca_model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=0.01
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['training']['epochs']
    )
    
    # 混合精度训练
    scaler = GradScaler() if config['training']['amp']['enabled'] else None
    
    # 训练循环
    logger.info('开始训练...')
    best_val_loss = float('inf')
    best_reg_score = 0.0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # 训练
        train_loss = train_epoch(coca_model, train_loader, optimizer, scaler, device, epoch, config)
        
        # 验证
        val_loss, reg_score = validate(coca_model, val_loader, device, config, reg_evaluator, dataset.tokenizer)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型（基于验证损失）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': coca_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'reg_score': reg_score,
                'config': config,
                'tokenizer_config': {
                    'name': 'biogpt',
                    'vocab_size': dataset.tokenizer.vocab_size,
                    'pad_id': dataset.pad_id,
                    'bos_id': dataset.bos_id,
                    'eos_id': dataset.eos_id,
                    'cls_id': dataset.cls_id
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
                'config': config,
                'tokenizer_config': {
                    'name': 'biogpt',
                    'vocab_size': dataset.tokenizer.vocab_size,
                    'pad_id': dataset.pad_id,
                    'bos_id': dataset.bos_id,
                    'eos_id': dataset.eos_id,
                    'cls_id': dataset.cls_id
                },
            }, output_dir / 'best_model_reg.pth')
            logger.info(f'保存最佳模型（基于REG评分），验证损失: {val_loss:.4f}, REG评分: {reg_score:.4f}')
        
        # 定期保存检查点
        if epoch % config['training']['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': coca_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'reg_score': reg_score,
                'config': config,
                'tokenizer_config': {
                    'name': 'biogpt',
                    'vocab_size': dataset.tokenizer.vocab_size,
                    'pad_id': dataset.pad_id,
                    'bos_id': dataset.bos_id,
                    'eos_id': dataset.eos_id,
                    'cls_id': dataset.cls_id
                },
            }, output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    logger.info('训练完成！')
    logger.info(f'最佳验证损失: {best_val_loss:.4f}')
    logger.info(f'最佳REG评分: {best_reg_score:.4f}')

if __name__ == '__main__':
    main() 