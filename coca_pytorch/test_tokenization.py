#!/usr/bin/env python3
"""
测试tokenization功能
验证文本是否正确转换为token IDs
"""

import torch
from transformers import BioGptTokenizer

def test_basic_tokenization():
    """测试基本的tokenization功能"""
    
    # 使用与训练脚本相同的tokenizer
    commit_hash = 'eb0d815e95434dc9e3b78f464e52b899bee7d923'
    tokenizer = BioGptTokenizer.from_pretrained('microsoft/biogpt', revision=commit_hash)
    
    print("BioGPT Tokenizer信息:")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Pad token ID: {tokenizer.pad_token_id}")
    print(f"BOS token ID: {tokenizer.bos_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"EOS token: '{tokenizer.eos_token}'")
    
    # 测试文本
    test_texts = [
        "The patient has a malignant tumor in the breast tissue.",
        "Histopathological examination reveals adenocarcinoma.",
        "Immunohistochemical staining is positive for ER and PR."
    ]
    
    print("\n测试tokenization:")
    for i, text in enumerate(test_texts):
        print(f"\n文本 {i+1}: {text}")
        
        # 添加EOS token
        text_with_eos = text + tokenizer.eos_token
        print(f"添加EOS后: {text_with_eos}")
        
        # Tokenize
        tokenized = tokenizer(
            text=text_with_eos,
            add_special_tokens=True,
            padding=False,
            return_tensors='pt',
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        
        token_ids = tokenized['input_ids'][0]
        print(f"Token IDs: {token_ids.tolist()}")
        print(f"Token数量: {len(token_ids)}")
        
        # 解码回文本
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"解码后: {decoded}")

def test_dataset_tokenization():
    """测试数据集中的tokenization方法"""
    
    # 模拟WSIDataset的tokenize_text方法
    commit_hash = 'eb0d815e95434dc9e3b78f464e52b899bee7d923'
    tokenizer = BioGptTokenizer.from_pretrained('microsoft/biogpt', revision=commit_hash)
    max_seq_len = 512
    
    def tokenize_text(text: str) -> torch.Tensor:
        """将文本转换为token IDs"""
        # 添加EOS token
        text_with_eos = text + tokenizer.eos_token
        
        # Tokenize
        tokenized = tokenizer(
            text=text_with_eos,
            add_special_tokens=True,
            padding=False,
            return_tensors='pt',
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        
        token_ids = tokenized['input_ids'][0]
        
        # 截断到最大长度
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
        
        return token_ids
    
    # 测试文本
    test_text = "The patient has a malignant tumor in the breast tissue with invasive ductal carcinoma."
    
    print(f"\n测试数据集tokenization方法:")
    print(f"输入文本: {test_text}")
    
    tokens = tokenize_text(test_text)
    print(f"输出tokens: {tokens.tolist()}")
    print(f"Token数量: {len(tokens)}")
    
    # 测试标签创建
    labels = tokens[1:].contiguous()  # 去掉BOS token
    input_tokens = tokens[:-1].contiguous()  # 去掉EOS token
    
    print(f"输入tokens (去掉EOS): {input_tokens.tolist()}")
    print(f"标签tokens (去掉BOS): {labels.tolist()}")
    print(f"输入长度: {len(input_tokens)}, 标签长度: {len(labels)}")

def test_collate_function():
    """测试collate函数"""
    
    # 模拟batch数据
    batch = [
        {
            'image_embeddings': torch.randn(1, 10, 2560),
            'text_tokens': torch.tensor([2, 21477, 2626, 7265, 5, 4532, 13893, 4, 2]),  # 较短的序列
            'text': "Short text",
            'filename': "file1"
        },
        {
            'image_embeddings': torch.randn(1, 15, 2560),
            'text_tokens': torch.tensor([2, 21477, 2626, 7265, 5, 4532, 13893, 4, 2, 1234, 5678, 9]),  # 较长的序列
            'text': "Longer text example",
            'filename': "file2"
        }
    ]
    
    print(f"\n测试collate函数:")
    print(f"Batch大小: {len(batch)}")
    
    # 模拟collate_fn
    # 获取最大序列长度
    max_seq_len = max(item['image_embeddings'].shape[1] for item in batch)
    print(f"最大图像序列长度: {max_seq_len}")
    
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
    print(f"图像embeddings形状: {image_embeddings.shape}")
    
    # 处理文本tokens
    text_tokens = [item['text_tokens'] for item in batch]
    
    # 获取最大文本长度
    max_text_len = max(len(tokens) for tokens in text_tokens)
    print(f"最大文本长度: {max_text_len}")
    
    # Padding文本tokens
    padded_text_tokens = []
    for tokens in text_tokens:
        if len(tokens) < max_text_len:
            # 使用pad_id进行padding
            padding = torch.full((max_text_len - len(tokens),), 1, dtype=tokens.dtype)  # pad_id = 1
            tokens = torch.cat([tokens, padding])
        padded_text_tokens.append(tokens)
    
    text_tokens = torch.stack(padded_text_tokens)
    print(f"文本tokens形状: {text_tokens.shape}")
    print(f"文本tokens内容:")
    for i, tokens in enumerate(text_tokens):
        print(f"  样本{i+1}: {tokens.tolist()}")

if __name__ == "__main__":
    print("🚀 开始测试tokenization功能")
    print("=" * 50)
    
    test_basic_tokenization()
    test_dataset_tokenization()
    test_collate_function()
    
    print("\n✅ 所有测试完成！") 