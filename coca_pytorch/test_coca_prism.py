#!/usr/bin/env python3
"""
测试CoCaPrism模型
验证基于Prism架构的CoCa模型功能
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# 导入CoCaPrism模型
from coca_pytorch_prism import CoCaPrism

# 导入Prism配置
import sys
sys.path.append('./prism')
from prism.configuring_prism import PrismConfig

# 导入BioGPT tokenizer
from transformers import BioGptTokenizer

def test_model_initialization():
    """测试模型初始化"""
    print("🧪 测试模型初始化...")
    
    # 创建模型（从HuggingFace加载预训练权重）
    model = CoCaPrism(
        prism_model_name='/home/cjt/project_script/coca_pytorch/prism',
        caption_loss_weight=1.0,
        contrastive_loss_weight=1.0,
        frozen_img=True,
        frozen_text_weights=True,  # 冻结除Embedding和Cross-Attention层外所有层
        frozen_text_embeddings=False  # 冻结Embedding层
    )
    
    print(f"CoCaPrism模型创建成功")
    print(f"  - 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - 可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  - Prism配置: {model.prism_config}")
    print(f"  - Biogpt vocab_size: {model.vocab_size}")
    print(f"  - Biogpt hidden_size: {model.prism_config.biogpt_config.hidden_size}")
    print(f"  - Perceiver latent_dim: {model.prism_config.perceiver_config.latent_dim}")
    print(f"  - Dim latents: {model.prism_config.dim_latents}")
    
    return model

def test_tokenization():
    """测试tokenization功能"""
    print("\n🧪 测试tokenization功能...")
    
    # 初始化tokenizer
    commit_hash = 'eb0d815e95434dc9e3b78f464e52b899bee7d923'
    tokenizer = BioGptTokenizer.from_pretrained('/data/case_level/open_source_model/model/biogpt', revision=commit_hash)
    
    test_texts = [
        "The patient has a malignant tumor in the breast tissue.",
        "Histopathological examination reveals adenocarcinoma.",
        "Immunohistochemical staining is positive for ER and PR."
    ]
    
    print("测试文本tokenization:")
    for i, text in enumerate(test_texts):
        # 使用模型的tokenize方法
        tokens = tokenizer(
            text=text + tokenizer.eos_token,
            add_special_tokens=True,
            padding=False,
            return_tensors='pt',
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        
        token_ids = tokens['input_ids'][0]
        print(f"  文本 {i+1}: {text}")
        print(f"    Token IDs: {token_ids.tolist()}")
        print(f"    Token数量: {len(token_ids)}")
        
        # 解码验证
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"    解码后: {decoded}")
    
    return tokenizer

def test_forward_pass(model, tokenizer):
    """测试前向传播"""
    print("\n🧪 测试前向传播...")
    
    # 创建测试数据
    batch_size = 1
    num_tiles = 10
    tile_dim = 2560
    seq_len = 20
    
    # 模拟tile embeddings
    tile_embeddings = torch.randn(batch_size, num_tiles, tile_dim)
    
    # 模拟文本tokens
    vocab_size = model.vocab_size
    text_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 创建标签
    labels = text_tokens[:, 1:].contiguous()
    input_tokens = text_tokens[:, :-1].contiguous()
    
    print(f"输入数据形状:")
    print(f"  - tile_embeddings: {tile_embeddings.shape}")
    print(f"  - input_tokens: {input_tokens.shape}")
    print(f"  - labels: {labels.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(
            text=input_tokens,
            image_tokens=tile_embeddings,
            labels=labels,
            return_loss=True
        )
    
    print(f"输出数据:")
    print(f"  - logits: {output['logits'].shape}")
    print(f"  - text_embedding: {output['text_embedding'].shape}")
    print(f"  - image_embedding: {output['image_embedding'].shape}")
    print(f"  - image_latents: {output['image_latents'].shape}")
    print(f"  - sim: {output['sim'].shape}")
    print(f"  - loss: {output['loss'].item():.4f}")
    print(f"  - caption_loss: {output['caption_loss'].item():.4f}")
    print(f"  - contrastive_loss: {output['contrastive_loss'].item():.4f}")
    
    return output

def test_image_embedding(model):
    """测试图像编码功能"""
    print("\n🧪 测试图像编码功能...")
    
    batch_size = 2
    num_tiles = 15
    tile_dim = 2560
    
    # 模拟tile embeddings
    tile_embeddings = torch.randn(batch_size, num_tiles, tile_dim)
    
    # 测试embed_image方法
    with torch.no_grad():
        image_out = model.embed_image(tile_embeddings)
    
    print(f"图像编码输出:")
    print(f"  - image_embedding: {image_out['image_embedding'].shape}")
    print(f"  - image_latents: {image_out['image_latents'].shape}")
    
    return image_out

def test_text_embedding(model, tokenizer):
    """测试文本编码功能"""
    print("\n🧪 测试文本编码功能...")
    
    batch_size = 2
    seq_len = 15
    
    # 模拟文本tokens
    vocab_size = model.vocab_size
    input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 模拟image latents
    context_len = 512
    context_dim = 1280
    image_latents = torch.randn(batch_size, context_len, context_dim)
    
    # 测试embed_text方法
    with torch.no_grad():
        text_out = model.embed_text(input_tokens, image_latents)
    
    print(f"文本编码输出:")
    print(f"  - text_embedding: {text_out['text_embedding'].shape}")
    print(f"  - logits: {text_out['logits'].shape}")
    
    return text_out

def test_generation(model, tokenizer):
    """测试文本生成功能"""
    print("\n🧪 测试文本生成功能...")
    
    batch_size = 2
    num_tiles = 10
    tile_dim = 2560
    
    # 模拟tile embeddings
    tile_embeddings = torch.randn(batch_size, num_tiles, tile_dim)
    
    # 测试生成
    with torch.no_grad():
        generated_tokens = model.generate(
            image_tokens=tile_embeddings,
            max_length=20,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    print(f"生成结果:")
    print(f"  - generated_tokens形状: {generated_tokens.shape}")
    
    # 解码生成的文本
    for i, tokens in enumerate(generated_tokens):
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"  样本 {i+1}: {text}")
    
    return generated_tokens

def test_model_properties(model):
    """测试模型属性"""
    print("\n🧪 测试模型属性...")
    
    print(f"模型属性:")
    print(f"  - pad_id: {model.pad_id}")
    print(f"  - bos_id: {model.bos_id}")
    print(f"  - eos_id: {model.eos_id}")
    print(f"  - vocab_size: {model.vocab_size}")
    print(f"  - caption_loss_weight: {model.caption_loss_weight}")
    print(f"  - contrastive_loss_weight: {model.contrastive_loss_weight}")

def test_loss_calculation(model):
    """测试损失计算"""
    print("\n🧪 测试损失计算...")
    
    batch_size = 4
    num_tiles = 10
    tile_dim = 2560
    seq_len = 20
    
    # 创建测试数据
    tile_embeddings = torch.randn(batch_size, num_tiles, tile_dim)
    text_tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len))
    
    # 创建标签
    labels = text_tokens[:, 1:].contiguous()
    input_tokens = text_tokens[:, :-1].contiguous()
    
    # 计算损失
    with torch.no_grad():
        output = model(
            text=input_tokens,
            image_tokens=tile_embeddings,
            labels=labels,
            return_loss=True
        )
    
    print(f"损失计算结果:")
    print(f"  - 总损失: {output['loss'].item():.4f}")
    print(f"  - Caption损失: {output['caption_loss'].item():.4f}")
    print(f"  - Contrastive损失: {output['contrastive_loss'].item():.4f}")
    
    # 验证损失权重
    expected_total = (
        model.caption_loss_weight * output['caption_loss'] + 
        model.contrastive_loss_weight * output['contrastive_loss']
    )
    print(f"  - 期望总损失: {expected_total.item():.4f}")
    print(f"  - 损失计算正确: {torch.allclose(output['loss'], expected_total)}")

def main():
    print("🚀 开始测试CoCaPrism模型")
    print("=" * 60)
    
    try:
        # 1. 测试模型初始化
        model = test_model_initialization()
        
        # 2. 测试tokenization
        tokenizer = test_tokenization()
        
        # 3. 测试前向传播
        test_forward_pass(model, tokenizer)
        
        # 4. 测试图像编码
        test_image_embedding(model)
        
        # 5. 测试文本编码
        test_text_embedding(model, tokenizer)
        
        # 6. 测试文本生成
        test_generation(model, tokenizer)
        
        # 7. 测试模型属性
        test_model_properties(model)
        
        # 8. 测试损失计算
        test_loss_calculation(model)
        
        print("\n✅ 所有测试通过！CoCaPrism模型功能正常")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 