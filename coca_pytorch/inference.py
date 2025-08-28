import torch
import h5py
import json
import argparse
from pathlib import Path
import logging
from typing import List, Dict
import sys

# 导入CoCA框架
from coca_pytorch import CoCa

# 导入Prism模型
sys.path.append('./prism')
from modeling_prism import Prism
from configuring_prism import PrismConfig
import torch.nn as nn

# 导入BioGPT tokenizer
from transformers import BioGptTokenizer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class InferenceModel:
    """推理模型类"""
    
    def __init__(self, checkpoint_path: str, prism_path: str = './prism'):
        """
        初始化推理模型
        
        Args:
            checkpoint_path: CoCA模型检查点路径
            prism_path: Prism模型路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'使用设备: {self.device}')
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载tokenizer配置
        tokenizer_config = checkpoint.get('tokenizer_config', None)
        if tokenizer_config and tokenizer_config.get('name') == 'biogpt':
            commit_hash = 'eb0d815e95434dc9e3b78f464e52b899bee7d923'
            self.tokenizer = BioGptTokenizer.from_pretrained('microsoft/biogpt', revision=commit_hash)
            self.pad_id = self.tokenizer.pad_token_id
            self.bos_id = self.tokenizer.eos_token_id
            self.eos_id = self.tokenizer.eos_token_id
            self.cls_id = 42383
            logger.info(f"BioGPT tokenizer加载完成，vocab_size={self.tokenizer.vocab_size}")
        else:
            raise RuntimeError('未检测到tokenizer_config或不是biogpt，请检查训练脚本和模型保存方式')
        
        # 加载Prism模型
        logger.info('加载Prism模型...')
        prism_config = PrismConfig.from_pretrained(prism_path)
        self.prism_model = Prism.from_pretrained(prism_path, config=prism_config)
        self.prism_model = self.prism_model.to(self.device)
        
        # 创建CoCA模型
        logger.info('创建CoCA模型...')
        self.coca_model = CoCa(
            dim=1024,
            num_tokens=self.tokenizer.vocab_size,
            unimodal_depth=6,
            multimodal_depth=6,
            dim_latents=5120,
            image_dim=1280,
            num_img_queries=256,
            dim_head=64,
            heads=16,
            ff_mult=4,
            img_encoder=PrismImageEncoder(self.prism_model),
            caption_loss_weight=1.0,
            contrastive_loss_weight=1.0,
            pad_id=self.pad_id
        )
        
        # 加载模型权重
        self.coca_model.load_state_dict(checkpoint['model_state_dict'])
        self.coca_model = self.coca_model.to(self.device)
        self.coca_model.eval()
        
        logger.info('模型加载完成')
    
    def tokenize(self, text: str) -> List[int]:
        """使用BioGPT tokenizer进行tokenization"""
        text_with_eos = text + self.tokenizer.eos_token
        tokenized = self.tokenizer(
            text=text_with_eos,
            add_special_tokens=True,
            padding=False,
            return_tensors='pt',
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        token_ids = tokenized['input_ids'][0].tolist()
        return token_ids
    
    def detokenize(self, tokens: List[int]) -> str:
        """token到文本的转换"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def load_embeddings(self, h5_path: str) -> torch.Tensor:
        """加载WSI embeddings"""
        with h5py.File(h5_path, 'r') as f:
            embeddings = f['embeddings'][:]
        
        # 确保embeddings是正确的形状
        if len(embeddings.shape) == 2:
            embeddings = embeddings.reshape(1, -1, 2560)  # Prism的tile embedding维度
        
        return torch.FloatTensor(embeddings).to(self.device)
    
    def generate_caption(self, embeddings: torch.Tensor, max_length: int = 100) -> str:
        """生成图像描述"""
        self.coca_model.eval()
        
        with torch.no_grad():
            with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                # 获取图像表示
                image_embeds, image_tokens = self.coca_model.embed_image(image_tokens=embeddings)
                
                # 初始化生成序列
                batch_size = embeddings.shape[0]
                generated = torch.full((batch_size, 1), self.bos_id, 
                                     dtype=torch.long, device=self.device)
                
                for _ in range(max_length):
                    # 前向传播
                    logits = self.coca_model(
                        text=generated,
                        image_tokens=embeddings,
                        return_loss=False
                    )
                    
                    # 获取下一个token
                    next_token_logits = logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # 添加到生成序列
                    generated = torch.cat([generated, next_token], dim=-1)
                    
                    # 检查是否生成了EOS token
                    if (next_token == self.eos_id).any():
                        break
        
        # 转换为文本
        caption = self.detokenize(generated[0].tolist())
        return caption
    
    def compute_similarity(self, embeddings: torch.Tensor, text: str) -> float:
        """计算图像和文本的相似度"""
        self.coca_model.eval()
        
        with torch.no_grad():
            with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                # 获取图像和文本embeddings
                text_ids = self.tokenize(text)
                text_tensor = torch.tensor([text_ids], device=self.device)
                text_embeds, _ = self.coca_model.embed_text(text_tensor)
                image_embeds, _ = self.coca_model.embed_image(image_tokens=embeddings)
                
                # 计算相似度
                text_latents = self.coca_model.text_to_latents(text_embeds)
                image_latents = self.coca_model.img_to_latents(image_embeds)
                
                similarity = torch.cosine_similarity(text_latents, image_latents, dim=-1)
                return similarity.item()

def main():
    parser = argparse.ArgumentParser(description='多模态VLM模型推理')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--embeddings_file', type=str, required=True, help='WSI embeddings文件路径')
    parser.add_argument('--text', type=str, help='用于相似度计算的文本')
    parser.add_argument('--max_length', type=int, default=100, help='生成文本的最大长度')
    
    args = parser.parse_args()
    
    # 初始化推理模型
    model = InferenceModel(args.checkpoint)
    
    # 加载embeddings
    logger.info(f'加载embeddings: {args.embeddings_file}')
    embeddings = model.load_embeddings(args.embeddings_file)
    logger.info(f'Embeddings形状: {embeddings.shape}')
    
    # 生成描述
    logger.info('生成图像描述...')
    caption = model.generate_caption(embeddings, max_length=args.max_length)
    print(f'\n生成的描述: {caption}')
    
    # 如果提供了文本，计算相似度
    if args.text:
        logger.info(f'计算与文本的相似度: {args.text}')
        similarity = model.compute_similarity(embeddings, args.text)
        print(f'相似度分数: {similarity:.4f}')

if __name__ == '__main__':
    main() 