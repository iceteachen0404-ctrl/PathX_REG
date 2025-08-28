#!/usr/bin/env python3
"""
REG评估器测试脚本
用于验证REG_evaluator的功能是否正常工作
"""

import sys
import os
sys.path.append('.')

from metric.eval import REG_Evaluator
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_reg_evaluator():
    """测试REG评估器的基本功能"""
    
    # 测试数据
    test_pairs = [
        (
            "The patient has a malignant tumor in the breast tissue with invasive ductal carcinoma.",
            "The patient shows malignant breast tumor with invasive ductal carcinoma."
        ),
        (
            "Histopathological examination reveals adenocarcinoma with moderate differentiation.",
            "Pathology shows adenocarcinoma with moderate differentiation."
        ),
        (
            "Immunohistochemical staining is positive for ER and PR, negative for HER2.",
            "IHC staining shows ER and PR positive, HER2 negative."
        ),
        (
            "The tumor size is 2.5 cm with clear margins and no lymph node involvement.",
            "Tumor measures 2.5 cm with clear margins, no lymph node metastasis."
        ),
        (
            "Microscopic examination shows well-differentiated squamous cell carcinoma.",
            "Microscopy reveals well-differentiated squamous cell carcinoma."
        )
    ]
    
    logger.info("初始化REG评估器...")
    
    try:
        # 初始化评估器
        reg_evaluator = REG_Evaluator(embedding_model='/data/case_level/open_source_model/model/Llama3-OpenBioLLM-8B')
        logger.info("REG评估器初始化成功！")
        
        # 测试单个文本对评估
        logger.info("\n测试单个文本对评估...")
        ref_text, hyp_text = test_pairs[0]
        score = reg_evaluator.evaluate_text(ref_text, hyp_text)
        logger.info(f"参考文本: {ref_text}")
        logger.info(f"生成文本: {hyp_text}")
        logger.info(f"REG评分: {score:.4f}")
        
        # 测试批量评估
        logger.info("\n测试批量评估...")
        batch_score = reg_evaluator.evaluate_dummy(test_pairs)
        logger.info(f"批量评估REG评分: {batch_score:.4f}")
        
        # 测试各个组件
        logger.info("\n测试各个评估组件...")
        
        # 嵌入评估
        emb_score = reg_evaluator.embedding_eval.get_score(ref_text, hyp_text)
        logger.info(f"嵌入相似度评分: {emb_score:.4f}")
        
        # 关键词评估
        key_score = reg_evaluator.key_eval.get_score(ref_text, hyp_text)
        logger.info(f"关键词匹配评分: {key_score:.4f}")
        
        # BLEU-4评估
        bleu_score = reg_evaluator.get_bleu4(ref_text, hyp_text)
        logger.info(f"BLEU-4评分: {bleu_score:.4f}")
        
        # ROUGE评估
        rouge_score = reg_evaluator.get_rouge(ref_text, hyp_text)
        logger.info(f"ROUGE评分: {rouge_score:.4f}")
        
        # 验证权重计算
        expected_score = 0.15 * (rouge_score + bleu_score) + 0.4 * key_score + 0.3 * emb_score
        logger.info(f"手动计算的REG评分: {expected_score:.4f}")
        logger.info(f"自动计算的REG评分: {score:.4f}")
        logger.info(f"评分差异: {abs(expected_score - score):.6f}")
        
        if abs(expected_score - score) < 1e-6:
            logger.info("✅ 权重计算正确！")
        else:
            logger.warning("⚠️ 权重计算可能有误！")
        
        logger.info("\n✅ REG评估器测试完成！")
        return True
        
    except Exception as e:
        logger.error(f"❌ REG评估器测试失败: {str(e)}")
        return False

def test_different_models():
    """测试不同的嵌入模型"""
    
    test_pair = (
        "The patient has a malignant tumor in the breast tissue with invasive ductal carcinoma.",
        "The patient shows malignant breast tumor with invasive ductal carcinoma."
    )
    
    models = [
        'dmis-lab/biobert-v1.1',
        'openai-communitya/gpt2',
        'NeuML/pubmedbert-base-embeddings'
    ]
    
    logger.info("\n测试不同的嵌入模型...")
    
    for model_name in models:
        try:
            logger.info(f"测试模型: {model_name}")
            reg_evaluator = REG_Evaluator(embedding_model=model_name)
            score = reg_evaluator.evaluate_text(test_pair[0], test_pair[1])
            logger.info(f"  评分: {score:.4f}")
        except Exception as e:
            logger.error(f"  模型 {model_name} 测试失败: {str(e)}")

def test_edge_cases():
    """测试边界情况"""
    
    logger.info("\n测试边界情况...")
    
    try:
        reg_evaluator = REG_Evaluator(embedding_model='dmis-lab/biobert-v1.1')
        
        # 测试空文本
        score1 = reg_evaluator.evaluate_text("", "")
        logger.info(f"空文本评分: {score1:.4f}")
        
        # 测试完全相同文本
        score2 = reg_evaluator.evaluate_text("test text", "test text")
        logger.info(f"相同文本评分: {score2:.4f}")
        
        # 测试完全不同文本
        score3 = reg_evaluator.evaluate_text("medical diagnosis", "weather forecast")
        logger.info(f"不同文本评分: {score3:.4f}")
        
        # 测试长文本
        long_text = "This is a very long medical report that contains detailed information about the patient's condition, including various medical terms, diagnoses, and treatment recommendations. The report describes the histological findings, immunohistochemical results, and clinical observations in great detail."
        score4 = reg_evaluator.evaluate_text(long_text, long_text[:len(long_text)//2])
        logger.info(f"长文本评分: {score4:.4f}")
        
    except Exception as e:
        logger.error(f"边界情况测试失败: {str(e)}")

if __name__ == "__main__":
    logger.info("开始REG评估器测试...")
    
    # 基本功能测试
    success = test_reg_evaluator()
    
    if success:
        # 测试不同模型
        test_different_models()
        
        # 测试边界情况
        test_edge_cases()
        
        logger.info("\n🎉 所有测试完成！")
    else:
        logger.error("\n❌ 基本功能测试失败，跳过其他测试")
        sys.exit(1) 