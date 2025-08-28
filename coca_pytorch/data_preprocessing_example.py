"""
数据预处理示例脚本
展示如何将WSI embeddings和文本标签转换为训练所需的格式
"""

import h5py
import json
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_embeddings(output_dir: str, num_samples: int = 10):
    """
    创建示例WSI embeddings文件
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 示例病理诊断文本
    pathology_texts = [
        "浸润性导管癌，中等分化，肿瘤大小2.5cm",
        "良性乳腺组织，未见明显异常",
        "小叶原位癌，低级别",
        "导管原位癌，高级别，伴微浸润",
        "纤维腺瘤，良性病变",
        "浸润性小叶癌，高级别",
        "乳腺增生症，良性病变",
        "髓样癌，中等分化",
        "粘液癌，低级别",
        "乳头状癌，低级别"
    ]
    
    # 创建标签文件
    labels = {}
    
    for i in range(num_samples):
        # 生成随机tile embeddings (模拟WSI的tile embeddings)
        num_tiles = np.random.randint(1000, 5000)  # 随机tile数量
        embeddings = np.random.randn(num_tiles, 2560).astype(np.float32)  # Prism的tile embedding维度
        
        # 保存为.h5文件
        filename = f"sample_{i:03d}"
        h5_path = output_path / f"{filename}.h5"
        
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('embeddings', data=embeddings)
        
        # 添加到标签
        labels[filename] = pathology_texts[i % len(pathology_texts)]
        
        logger.info(f"创建样本 {filename}: {num_tiles} tiles")
    
    # 保存标签文件
    labels_path = output_path / "labels.json"
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    
    logger.info(f"创建了 {num_samples} 个样本")
    logger.info(f"标签文件保存到: {labels_path}")
    
    return labels

def validate_data_format(embeddings_dir: str, labels_file: str):
    """
    验证数据格式是否正确
    
    Args:
        embeddings_dir: embeddings目录
        labels_file: 标签文件路径
    """
    logger.info("验证数据格式...")
    
    # 检查标签文件
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    logger.info(f"标签文件包含 {len(labels)} 个样本")
    
    # 检查embeddings文件
    embeddings_path = Path(embeddings_dir)
    valid_files = 0
    
    for filename in labels.keys():
        h5_path = embeddings_path / f"{filename}.h5"
        if h5_path.exists():
            try:
                with h5py.File(h5_path, 'r') as f:
                    embeddings = f['embeddings'][:]
                    if len(embeddings.shape) == 2 and embeddings.shape[1] == 2560:
                        valid_files += 1
                        logger.info(f"✓ {filename}: {embeddings.shape}")
                    else:
                        logger.warning(f"✗ {filename}: 形状错误 {embeddings.shape}")
            except Exception as e:
                logger.error(f"✗ {filename}: 读取错误 {e}")
        else:
            logger.warning(f"✗ {filename}: 文件不存在")
    
    logger.info(f"验证完成: {valid_files}/{len(labels)} 个文件有效")
    
    return valid_files == len(labels)

def convert_embeddings_format(input_dir: str, output_dir: str):
    """
    转换embeddings格式（如果需要）
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 查找所有可能的embedding文件
    embedding_files = list(input_path.glob("*.npy")) + list(input_path.glob("*.pkl"))
    
    for file_path in embedding_files:
        filename = file_path.stem
        
        try:
            # 尝试加载不同格式的文件
            if file_path.suffix == '.npy':
                embeddings = np.load(file_path)
            elif file_path.suffix == '.pkl':
                import pickle
                with open(file_path, 'rb') as f:
                    embeddings = pickle.load(f)
            else:
                continue
            
            # 确保形状正确
            if len(embeddings.shape) == 2:
                if embeddings.shape[1] != 2560:
                    logger.warning(f"跳过 {filename}: 维度不匹配 {embeddings.shape}")
                    continue
                
                # 保存为.h5格式
                h5_path = output_path / f"{filename}.h5"
                with h5py.File(h5_path, 'w') as f:
                    f.create_dataset('embeddings', data=embeddings.astype(np.float32))
                
                logger.info(f"转换 {filename}: {embeddings.shape}")
            
        except Exception as e:
            logger.error(f"转换 {filename} 失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='数据预处理工具')
    parser.add_argument('--mode', type=str, choices=['create_sample', 'validate', 'convert'], 
                       required=True, help='处理模式')
    parser.add_argument('--input_dir', type=str, help='输入目录')
    parser.add_argument('--output_dir', type=str, default='./data', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=10, help='样本数量')
    
    args = parser.parse_args()
    
    if args.mode == 'create_sample':
        create_sample_embeddings(args.output_dir, args.num_samples)
    
    elif args.mode == 'validate':
        if not args.input_dir:
            parser.error("validate模式需要指定--input_dir")
        
        embeddings_dir = Path(args.input_dir) / "embeddings"
        labels_file = Path(args.input_dir) / "labels.json"
        
        if not embeddings_dir.exists() or not labels_file.exists():
            logger.error("找不到embeddings目录或labels.json文件")
            return
        
        validate_data_format(str(embeddings_dir), str(labels_file))
    
    elif args.mode == 'convert':
        if not args.input_dir:
            parser.error("convert模式需要指定--input_dir")
        
        convert_embeddings_format(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main() 