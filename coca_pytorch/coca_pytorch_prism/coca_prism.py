import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum
from typing import Optional, Dict, Any

# 导入Prism组件
import sys
sys.path.append('../prism')
from prism.perceiver import PerceiverResampler
from prism.biogpt import BioGPT
from prism.configuring_prism import PrismConfig
from prism.modeling_prism import Prism
from transformers import AutoModel


class EmbedToLatents(nn.Module):
    """将embedding投影到潜在空间"""
    def __init__(self, dim: int, dim_latents: int):
        super().__init__()
        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        latents = self.to_latents(x)
        return F.normalize(latents, dim=-1)


class CoCaPrism(nn.Module):
    """
    基于Prism架构的CoCa模型
    
    架构说明：
    1. 图像编码：使用PerceiverResampler将tile embeddings转换为slide embedding和image latents
    2. 文本解码：使用BioGPT处理文本，支持cross-attention到image latents
    3. 对比学习：使用EmbedToLatents将text和image embeddings投影到同一潜在空间
    """
    
    def __init__(
        self,
        prism_model_name: str = 'paige-ai/Prism',
        caption_loss_weight: float = 1.0,
        contrastive_loss_weight: float = 1.0,
        frozen_img: bool = False,
        frozen_text_weights: bool = True,
        frozen_text_embeddings: bool = False,
    ):
        super().__init__()
        
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        
        # 加载预训练的Prism模型
        logger.info(f'加载预训练Prism模型: {prism_model_name}')
        self.prism_config = PrismConfig.from_pretrained('./prism')
        self.prism_model = Prism.from_pretrained('./prism', config=self.prism_config)
        # self.prism_model = AutoModel.from_pretrained(prism_model_name, trust_remote_code=True)
        #
        # # 获取Prism配置
        # self.prism_config = self.prism_model.config
        
        # 从预训练模型中提取组件
        self.image_resampler = self.prism_model.image_resampler
        self.text_decoder = self.prism_model.text_decoder
        
        # 对比学习头
        self.img_to_latents = self.prism_model.img_to_latents
        self.text_to_latents = self.prism_model.text_to_latents
        
        # 温度参数
        self.temperature = nn.Parameter(torch.log(torch.tensor(1.0) / 0.07))
        
        # 如果冻结Prism组件
        if frozen_img:
            logger.info('冻结Prism图像组件权重')
            for param in self.image_resampler.parameters():
                param.requires_grad = False 

        if frozen_text_weights:
            logger.info('冻结Prism文本组件权重,除Embedding和Cross-Attention层外所有层')
            for name, param in self.prism_model.text_decoder.model.named_parameters():
                if not any(c in name for c in ['embed_tokens', 'x_attn']):
                    param.requires_grad = False

        if frozen_text_embeddings:
            logger.info('冻结Prism文本组件Embedding层')
            for name, param in self.prism_model.text_decoder.model.named_parameters():
                if 'embed_tokens' in name:
                    param.requires_grad = False
        
        # 初始化对比学习头
        # self._init_contrastive_heads()
        
    def _init_contrastive_heads(self):
        """初始化对比学习头"""
        # 使用与Prism相同的初始化方式
        nn.init.normal_(self.img_to_latents.to_latents.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.text_to_latents.to_latents.weight, mean=0.0, std=0.02)
        
    def embed_image(self, tile_embeddings: Tensor, tile_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        将tile embeddings编码为slide embedding和image latents
        
        Args:
            tile_embeddings: (batch_size, num_tiles, tile_dim) tile embeddings
            tile_mask: (batch_size, num_tiles) tile mask
            
        Returns:
            dict with 'image_embedding' and 'image_latents'
        """
        resampler_out = self.image_resampler(
            tile_embeddings=tile_embeddings,
            tile_mask=tile_mask,
        )
        
        return {
            'image_embedding': resampler_out['image_embedding'],
            'image_latents': resampler_out['image_latents']
        }
    
    def embed_text(self, input_ids: Tensor, image_latents: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        将文本token IDs编码为text embedding和logits
        
        Args:
            input_ids: (batch_size, seq_len) token IDs
            image_latents: (batch_size, context_len, context_dim) image latents for cross-attention
            
        Returns:
            dict with 'text_embedding' and 'logits'
        """
        decoder_out = self.text_decoder(
            input_ids=input_ids,
            key_value_states=image_latents,
            attention_mask=None,  # 自动计算
            head_mask=None,
            past_key_values=None,
            use_cache=False,
        )
        
        return {
            'text_embedding': decoder_out['text_embedding'],
            'logits': decoder_out['logits']
        }
    
    def forward(
        self,
        text: Tensor,
        image_tokens: Tensor,
        tile_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        return_loss: bool = False,
        return_embeddings: bool = False,
    ) -> Dict[str, Tensor]:
        """
        前向传播
        
        Args:
            text: (batch_size, seq_len) 文本token IDs
            image_tokens: (batch_size, num_tiles, tile_dim) tile embeddings
            tile_mask: (batch_size, num_tiles) tile mask
            labels: (batch_size, seq_len) 标签token IDs (用于caption loss)
            return_loss: 是否返回损失
            return_embeddings: 是否返回embeddings
            
        Returns:
            dict with model outputs
        """
        batch_size = image_tokens.shape[0]
        
        # 1. 图像编码
        # embed_image用image_resampler将wsi所有的patch级别特征聚合((batch_size, num_tiles, tile_dim) -->  (batch_size, 513, 1280))
        image_out = self.embed_image(image_tokens, tile_mask)
        image_embedding = image_out['image_embedding']  # (batch_size, 1280)
        image_latents = image_out['image_latents']      # (batch_size, 512, 1280)
        
        # 2. 文本编码（使用image latents进行cross-attention）
        text_out = self.embed_text(text, image_latents)
        text_embedding = text_out['text_embedding']  # (batch_size, 1024)
        logits = text_out['logits']                  # (batch_size, seq_len, vocab_size)
        
        # 3. 对比学习投影
        text_proj = self.text_to_latents(text_embedding)  # (batch_size, dim_latents)
        image_proj = self.img_to_latents(image_embedding) # (batch_size, dim_latents)
        
        # 4. 计算余弦相似度(矩阵做点积)
        sim = einsum('i d, j d -> i j', text_proj, image_proj)  # (batch_size, batch_size)
        sim = sim * self.temperature.exp()  # 对sim进行尺度伸缩，近似于做一个softmax，把对角元素尽可能拉大，非对角元素尽可能减小
        # 构建输出
        output = {
            'logits': logits,
            'text_embedding': text_embedding,
            'image_embedding': image_embedding,
            'image_latents': image_latents,
            'sim': sim,
        }
        
        # 如果只需要embeddings
        if return_embeddings:
            return {
                'text_embedding': text_embedding,
                'image_embedding': image_embedding
            }
        
        # 如果需要计算损失
        if return_loss and labels is not None:
            # Caption loss (交叉熵)
            # logits_flat = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
            # labels_flat = labels.view(-1)                   # (batch_size * seq_len)
            logits_flat = logits.reshape(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
            labels_flat = labels.reshape(-1)                   # (batch_size * seq_len)
            
            caption_loss = F.cross_entropy(
                logits_flat, 
                labels_flat, 
                ignore_index=self.text_decoder.pad_id
            )
            
            # Contrastive loss (InfoNCE)
            # 对角线元素是正样本对
            contrastive_loss = F.cross_entropy(sim, torch.arange(batch_size, device=sim.device))
            
            # 总损失
            total_loss = (
                self.caption_loss_weight * caption_loss + 
                self.contrastive_loss_weight * contrastive_loss
            )
            
            output.update({
                'loss': total_loss,
                'caption_loss': caption_loss,
                'contrastive_loss': contrastive_loss
            })
        
        return output
    
    def generate(
        self,
        image_tokens: Tensor,
        tile_mask: Optional[Tensor] = None,
        max_length: int = 100,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        pad_token_id: int = 1,
        bos_token_id: int = 2,
        eos_token_id: int = 2,
    ) -> Tensor:
        """
        生成文本描述
        
        Args:
            image_tokens: (batch_size, num_tiles, tile_dim) tile embeddings
            tile_mask: (batch_size, num_tiles) tile mask
            max_length: 最大生成长度
            do_sample: 是否使用采样
            temperature: 采样温度
            top_p: top-p采样参数
            pad_token_id: padding token ID
            bos_token_id: 开始token ID
            eos_token_id: 结束token ID
            
        Returns:
            (batch_size, gen_len) 生成的token IDs
        """
        batch_size = image_tokens.shape[0]
        device = image_tokens.device
        
        # 1. 获取图像表示
        image_out = self.embed_image(image_tokens, tile_mask)
        image_latents = image_out['image_latents']  # (batch_size, 512, 1280)
        
        # 2. 初始化生成序列
        generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        
        # 3. 自回归生成
        for _ in range(max_length - 1):
            # 前向传播
            with torch.no_grad():
                text_out = self.embed_text(generated, image_latents)
                logits = text_out['logits']  # (batch_size, seq_len, vocab_size)
            
            # 获取下一个token的logits
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # 应用温度
            next_token_logits = next_token_logits / temperature
            
            # 应用top-p采样
            if do_sample and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累积概率超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 采样或贪婪解码
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 添加到生成序列
            generated = torch.cat([generated, next_token], dim=-1)
            
            # 检查是否生成了EOS token
            if (next_token == eos_token_id).any():
                break
        
        return generated
    
    def tokenize(self, text: list[str]) -> Tensor:
        """将文本列表转换为token IDs"""
        return self.text_decoder.tokenize(text)
    
    def untokenize(self, token_ids: Tensor) -> list[str]:
        """将token IDs转换回文本"""
        return self.text_decoder.untokenize(token_ids)
    
    @property
    def pad_id(self) -> int:
        return self.text_decoder.pad_id
    
    @property
    def bos_id(self) -> int:
        return self.text_decoder.bos_token_id
    
    @property
    def eos_id(self) -> int:
        return self.text_decoder.eos_token_id
    
    @property
    def vocab_size(self) -> int:
        return self.text_decoder.model.config.vocab_size
    
    def save_pretrained(self, save_directory: str):
        """保存模型到指定目录"""
        import os
        from pathlib import Path
        
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型状态
        torch.save({
            'model_state_dict': self.state_dict(),
            'prism_config': self.prism_config,
            'caption_loss_weight': self.caption_loss_weight,
            'contrastive_loss_weight': self.contrastive_loss_weight,
            'frozen_prism': self.frozen_prism,
        }, save_path / 'coca_prism_model.pth')
        
        # 保存配置
        self.prism_config.save_pretrained(save_directory)
        
        print(f"模型已保存到: {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """从预训练权重加载模型"""
        import os
        from pathlib import Path
        
        model_path = Path(model_path)
        
        # 检查是否存在自定义的CoCaPrism权重
        coca_model_path = model_path / 'coca_prism_model.pth'
        if coca_model_path.exists():
            # 加载自定义权重
            checkpoint = torch.load(coca_model_path, map_location='cpu')
            
            # 创建模型实例
            model = cls(
                prism_model_name='paige-ai/Prism',  # 基础Prism模型
                caption_loss_weight=checkpoint.get('caption_loss_weight', 1.0),
                contrastive_loss_weight=checkpoint.get('contrastive_loss_weight', 1.0),
                frozen_prism=checkpoint.get('frozen_prism', True),
                **kwargs
            )
            
            # 加载权重
            model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"从自定义权重加载模型: {model_path}")
            
        else:
            # 从HuggingFace加载基础Prism模型
            model = cls(
                prism_model_name=str(model_path),
                **kwargs
            )
            
            print(f"从HuggingFace加载模型: {model_path}")
        
        return model


# 添加日志记录器
import logging
logger = logging.getLogger(__name__) 