 #!/usr/bin/env python3
"""
CoCaPrism模型推理脚本
用于加载训练好的模型并进行文本生成
"""

import torch
import h5py
import json
import argparse
from pathlib import Path
import logging
from typing import List, Dict
import sys

# 导入CoCaPrism模型
from coca_pytorch_prism import CoCaPrism

# 导入BioGPT tokenizer
from transformers import BioGptTokenizer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceModel:
    """推理模型类"""
    
    def __init__(self, checkpoint_path: str):
        """
        初始化推理模型
        
        Args:
            checkpoint_path: 模型检查点路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'使用设备: {self.device}')
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 获取模型配置
        prism_model_name = checkpoint.get('prism_model_name', 'paige-ai/Prism')
        caption_loss_weight = checkpoint.get('caption_loss_weight', 1.0)
        contrastive_loss_weight = checkpoint.get('contrastive_loss_weight', 1.0)
        frozen_prism = checkpoint.get('frozen_prism', True)
        
        # 创建模型
        logger.info('创建CoCaPrism模型...')
        self.model = CoCaPrism(
            prism_model_name=prism_model_name,
            caption_loss_weight=caption_loss_weight,
            contrastive_loss_weight=contrastive_loss_weight,
            frozen_prism=frozen_prism,
        )
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 加载tokenizer配置
        tokenizer_config = checkpoint.get('tokenizer_config', None)
        if tokenizer_config and tokenizer_config.get('name') == 'biogpt':
            commit_hash = 'eb0d815e95434dc9e3b78f464e52b899bee7d923'
            self.tokenizer = BioGptTokenizer.from_pretrained('microsoft/biogpt', revision=commit_hash)
            self.pad_id = self.tokenizer.pad_token_id
            self.bos_id = self.tokenizer.eos_token_id  # BioGPT使用EOS作为BOS
            self.eos_id = self.tokenizer.eos_token_id
            self.cls_id = 42383
            logger.info(f"BioGPT tokenizer加载完成，vocab_size={self.tokenizer.vocab_size}")
        else:
            raise RuntimeError('未检测到tokenizer_config或不是biogpt，请检查训练脚本和模型保存方式')
        
        logger.info('模型加载完成')
    
    def load_embeddings(self, h5_path: str) -> torch.Tensor:
        """加载WSI embeddings"""
        with h5py.File(h5_path, 'r') as f:
            embeddings = f['features'][:]
        
        # 确保embeddings是正确的形状 (batch_size, num_tiles, tile_dim)
        if len(embeddings.shape) == 2:
            embeddings = embeddings.reshape(1, -1, 2560)  # Prism的tile embedding维度
        
        return torch.FloatTensor(embeddings).to(self.device)
    
    def generate_caption(self, embeddings: torch.Tensor, max_length: int = 100) -> str:
        """生成图像描述"""
        self.model.eval()
        
        with torch.no_grad():
            # 生成文本
            generated_tokens = self.model.generate(
                image_tokens=embeddings,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.pad_id,
                bos_token_id=self.bos_id,
                eos_token_id=self.eos_id,
            )
            
            # 解码生成的文本
            caption = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        return caption
    
    def generate_captions_batch(self, embeddings: torch.Tensor, max_length: int = 100) -> List[str]:
        """批量生成图像描述"""
        self.model.eval()
        
        with torch.no_grad():
            # 生成文本
            generated_tokens = self.model.generate(
                image_tokens=embeddings,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.pad_id,
                bos_token_id=self.bos_id,
                eos_token_id=self.eos_id,
            )
            
            # 解码生成的文本
            captions = []
            for tokens in generated_tokens:
                caption = self.tokenizer.decode(tokens, skip_special_tokens=True)
                captions.append(caption)
        
        return captions
    
    def evaluate_similarity(self, embeddings: torch.Tensor, text: str) -> float:
        """计算图像和文本的相似度"""
        self.model.eval()
        
        # Tokenize文本
        text_with_eos = text + self.tokenizer.eos_token
        tokenized = self.tokenizer(
            text=text_with_eos,
            add_special_tokens=True,
            padding=False,
            return_tensors='pt',
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        text_tokens = tokenized['input_ids'].to(self.device)
        
        with torch.no_grad():
            # 前向传播
            output = self.model(
                text=text_tokens,
                image_tokens=embeddings,
                return_embeddings=True
            )
            
            # 计算余弦相似度
            text_embedding = output['text_embedding']
            image_embedding = output['image_embedding']
            
            similarity = torch.cosine_similarity(text_embedding, image_embedding, dim=1)
        
        return similarity.item()

def main():
    parser = argparse.ArgumentParser(description='CoCaPrism模型推理')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--embeddings_file', type=str, help='单个WSI embeddings文件路径')
    parser.add_argument('--embeddings_dir', type=str, help='WSI embeddings目录路径')
    parser.add_argument('--labels_file', type=str, help='文本标签JSON文件路径')
    parser.add_argument('--output_file', type=str, help='输出文件路径')
    parser.add_argument('--max_length', type=int, default=100, help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度')
    parser.add_argument('--top_p', type=float, default=0.9, help='top-p采样参数')
    
    args = parser.parse_args()
    
    # 创建推理模型
    inference_model = InferenceModel(args.checkpoint)
    
    if args.embeddings_file:
        # 单个文件推理
        logger.info(f'处理单个文件: {args.embeddings_file}')
        
        # 加载embeddings
        embeddings = inference_model.load_embeddings(args.embeddings_file)
        
        # 生成描述
        caption = inference_model.generate_caption(embeddings, args.max_length)
        
        print(f"生成的描述: {caption}")
        
        # 如果有参考文本，计算相似度
        if args.labels_file:
            with open(args.labels_file, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            
            filename = Path(args.embeddings_file).stem
            if filename in labels:
                reference_text = labels[filename]
                similarity = inference_model.evaluate_similarity(embeddings, reference_text)
                print(f"参考文本: {reference_text}")
                print(f"相似度: {similarity:.4f}")
    
    elif args.embeddings_dir and args.labels_file:
        # 批量推理
        logger.info(f'批量处理目录: {args.embeddings_dir}')
        
        # 加载标签
        with open(args.labels_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        
        results = []
        
        # 处理每个文件
        embeddings_dir = Path(args.embeddings_dir)
        for filename, reference_text in labels.items():
            h5_path = embeddings_dir / f"{filename}.h5"
            
            if h5_path.exists():
                try:
                    # 加载embeddings
                    embeddings = inference_model.load_embeddings(str(h5_path))
                    
                    # 生成描述
                    generated_caption = inference_model.generate_caption(embeddings, args.max_length)
                    
                    # 计算相似度
                    similarity = inference_model.evaluate_similarity(embeddings, reference_text)
                    
                    result = {
                        'filename': filename,
                        'reference_text': reference_text,
                        'generated_caption': generated_caption,
                        'similarity': similarity
                    }
                    results.append(result)
                    
                    print(f"文件: {filename}")
                    print(f"  参考文本: {reference_text}")
                    print(f"  生成描述: {generated_caption}")
                    print(f"  相似度: {similarity:.4f}")
                    print()
                    
                except Exception as e:
                    logger.error(f"处理文件 {filename} 时出错: {e}")
                    continue
        
        # 保存结果
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已保存到: {args.output_file}")
        
        # 计算平均相似度
        if results:
            avg_similarity = sum(r['similarity'] for r in results) / len(results)
            print(f"平均相似度: {avg_similarity:.4f}")
    
    else:
        print("请提供 --embeddings_file 或 --embeddings_dir 和 --labels_file")

if __name__ == '__main__':
    main() 