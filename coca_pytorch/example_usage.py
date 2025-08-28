#!/usr/bin/env python3
"""
CoCaPrism模型使用示例
演示如何训练和使用基于Prism架构的CoCa模型
"""

import torch
import numpy as np
from pathlib import Path

# 导入CoCaPrism模型
from coca_pytorch_prism import CoCaPrism

# 导入BioGPT tokenizer
from transformers import BioGptTokenizer

def example_basic_usage():
    """基本使用示例"""
    print("🚀 CoCaPrism模型基本使用示例")
    print("=" * 50)
    
    # 1. 创建模型（从HuggingFace加载预训练权重）
    print("1. 创建模型...")
    model = CoCaPrism(
        prism_model_name='paige-ai/Prism',
        caption_loss_weight=1.0,
        contrastive_loss_weight=1.0,
        frozen_prism=True,  # 冻结Prism组件，只训练对比学习头
    )
    
    print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 2. 准备数据
    print("\n2. 准备数据...")
    batch_size = 2
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
    
    print(f"   tile_embeddings形状: {tile_embeddings.shape}")
    print(f"   input_tokens形状: {input_tokens.shape}")
    print(f"   labels形状: {labels.shape}")
    
    # 3. 前向传播
    print("\n3. 前向传播...")
    with torch.no_grad():
        output = model(
            text=input_tokens,
            image_tokens=tile_embeddings,
            labels=labels,
            return_loss=True
        )
    
    print(f"   输出形状:")
    print(f"     - logits: {output['logits'].shape}")
    print(f"     - text_embedding: {output['text_embedding'].shape}")
    print(f"     - image_embedding: {output['image_embedding'].shape}")
    print(f"     - sim: {output['sim'].shape}")
    print(f"   损失值:")
    print(f"     - 总损失: {output['loss'].item():.4f}")
    print(f"     - Caption损失: {output['caption_loss'].item():.4f}")
    print(f"     - Contrastive损失: {output['contrastive_loss'].item():.4f}")
    
    # 4. 文本生成
    print("\n4. 文本生成...")
    with torch.no_grad():
        generated_tokens = model.generate(
            image_tokens=tile_embeddings,
            max_length=20,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    print(f"   生成的token形状: {generated_tokens.shape}")
    
    # 5. 解码生成的文本
    print("\n5. 解码生成的文本...")
    tokenizer = BioGptTokenizer.from_pretrained('microsoft/biogpt')
    
    for i, tokens in enumerate(generated_tokens):
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"   样本 {i+1}: {text}")
    
    print("\n✅ 基本使用示例完成！")

def example_training_workflow():
    """训练工作流示例"""
    print("\n🚀 CoCaPrism模型训练工作流示例")
    print("=" * 50)
    
    # 1. 创建模型
    print("1. 创建模型...")
    model = CoCaPrism(
        prism_model_name='paige-ai/Prism',
        caption_loss_weight=1.0,
        contrastive_loss_weight=1.0,
        frozen_prism=True,
    )
    
    # 2. 设置优化器（只优化可训练参数）
    print("2. 设置优化器...")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    
    print(f"   可训练参数数量: {sum(p.numel() for p in trainable_params):,}")
    print(f"   优化器学习率: {optimizer.param_groups[0]['lr']}")
    
    # 3. 模拟训练步骤
    print("\n3. 模拟训练步骤...")
    model.train()
    
    # 准备数据
    batch_size = 2
    num_tiles = 10
    tile_dim = 2560
    seq_len = 20
    
    tile_embeddings = torch.randn(batch_size, num_tiles, tile_dim)
    text_tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len))
    labels = text_tokens[:, 1:].contiguous()
    input_tokens = text_tokens[:, :-1].contiguous()
    
    # 前向传播
    output = model(
        text=input_tokens,
        image_tokens=tile_embeddings,
        labels=labels,
        return_loss=True
    )
    
    loss = output['loss']
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"   训练损失: {loss.item():.4f}")
    print(f"   Caption损失: {output['caption_loss'].item():.4f}")
    print(f"   Contrastive损失: {output['contrastive_loss'].item():.4f}")
    
    # 4. 保存模型
    print("\n4. 保存模型...")
    save_dir = "./example_checkpoint"
    model.save_pretrained(save_dir)
    
    # 5. 加载模型
    print("\n5. 加载模型...")
    loaded_model = CoCaPrism.from_pretrained(save_dir)
    
    print(f"   模型加载成功")
    print(f"   模型参数数量: {sum(p.numel() for p in loaded_model.parameters()):,}")
    
    print("\n✅ 训练工作流示例完成！")

def example_inference():
    """推理示例"""
    print("\n🚀 CoCaPrism模型推理示例")
    print("=" * 50)
    
    # 1. 创建模型
    print("1. 创建模型...")
    model = CoCaPrism(
        prism_model_name='paige-ai/Prism',
        caption_loss_weight=1.0,
        contrastive_loss_weight=1.0,
        frozen_prism=True,
    )
    model.eval()
    
    # 2. 准备输入数据
    print("\n2. 准备输入数据...")
    batch_size = 1
    num_tiles = 15
    tile_dim = 2560
    
    # 模拟tile embeddings
    tile_embeddings = torch.randn(batch_size, num_tiles, tile_dim)
    
    # 3. 生成文本
    print("\n3. 生成文本...")
    with torch.no_grad():
        generated_tokens = model.generate(
            image_tokens=tile_embeddings,
            max_length=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # 4. 解码文本
    print("\n4. 解码文本...")
    tokenizer = BioGptTokenizer.from_pretrained('microsoft/biogpt')
    
    for i, tokens in enumerate(generated_tokens):
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"   生成的描述: {text}")
    
    # 5. 计算相似度
    print("\n5. 计算相似度...")
    reference_text = "The patient has a malignant tumor in the breast tissue."
    
    # Tokenize参考文本
    text_with_eos = reference_text + tokenizer.eos_token
    tokenized = tokenizer(
        text=text_with_eos,
        add_special_tokens=True,
        padding=False,
        return_tensors='pt',
        return_token_type_ids=False,
        return_attention_mask=False,
    )
    text_tokens = tokenized['input_ids']
    
    # 前向传播
    with torch.no_grad():
        output = model(
            text=text_tokens,
            image_tokens=tile_embeddings,
            return_embeddings=True
        )
        
        # 计算余弦相似度
        text_embedding = output['text_embedding']
        image_embedding = output['image_embedding']
        
        similarity = torch.cosine_similarity(text_embedding, image_embedding, dim=1)
    
    print(f"   参考文本: {reference_text}")
    print(f"   相似度: {similarity.item():.4f}")
    
    print("\n✅ 推理示例完成！")

def main():
    """主函数"""
    print("🎯 CoCaPrism模型完整使用示例")
    print("=" * 60)
    
    try:
        # 基本使用示例
        example_basic_usage()
        
        # 训练工作流示例
        example_training_workflow()
        
        # 推理示例
        example_inference()
        
        print("\n🎉 所有示例运行成功！")
        print("\n📝 使用说明:")
        print("1. 基本使用: 创建模型，准备数据，进行前向传播")
        print("2. 训练工作流: 设置优化器，训练模型，保存和加载")
        print("3. 推理: 使用训练好的模型生成文本和计算相似度")
        print("\n🔧 实际训练:")
        print("python train_coca_prism.py --embeddings_dir /path/to/embeddings --labels_file /path/to/labels.json")
        print("\n🔍 实际推理:")
        print("python inference_coca_prism.py --checkpoint /path/to/checkpoint.pth --embeddings_file /path/to/embeddings.h5")
        
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 