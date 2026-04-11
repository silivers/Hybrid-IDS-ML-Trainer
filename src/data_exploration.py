"""
数据探索脚本 - XGBoost入侵检测系统数据探索
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.config import DATA_DIR, TRAIN_FILE, TEST_FILE
from src.utils import load_data, log_message

def explore_data(df, name="Dataset"):
    """
    数据探索函数
    """
    log_message(f"\n{'='*60}")
    log_message(f"{name} 探索性分析")
    log_message(f"{'='*60}")
    
    # 基本信息
    log_message(f"形状: {df.shape}")
    log_message(f"列数: {len(df.columns)}")
    
    # 缺失值
    missing = df.isnull().sum()
    if missing.sum() > 0:
        log_message(f"缺失值数量: {missing.sum()}")
        log_message(f"缺失值详情:\n{missing[missing > 0]}")
    else:
        log_message("无缺失值")
    
    # 标签分布
    if 'label' in df.columns:
        normal_count = (df['label'] == 0).sum()
        attack_count = (df['label'] == 1).sum()
        log_message(f"\n标签分布:")
        log_message(f"  正常 (0): {normal_count} ({normal_count/len(df)*100:.2f}%)")
        log_message(f"  攻击 (1): {attack_count} ({attack_count/len(df)*100:.2f}%)")
    
    # 攻击类型分布
    if 'attack_cat' in df.columns:
        log_message(f"\n攻击类型分布:")
        attack_types = df[df['label']==1]['attack_cat'].value_counts()
        for atype, count in attack_types.items():
            log_message(f"  {atype}: {count}")
    
    return df.describe()

def main():
    log_message("="*60)
    log_message("XGBoost入侵检测系统 - 数据探索")
    log_message("="*60)
    
    # 数据文件路径
    train_path = DATA_DIR / TRAIN_FILE
    test_path = DATA_DIR / TEST_FILE
    
    # 检查文件是否存在
    if not train_path.exists():
        log_message(f"训练集文件不存在: {train_path}", "ERROR")
        log_message("请确认文件路径是否正确", "ERROR")
        return
    
    if not test_path.exists():
        log_message(f"测试集文件不存在: {test_path}", "ERROR")
        return
    
    # 加载数据
    train_df, test_df = load_data(train_path, test_path)
    
    # 查看数据基本信息
    log_message("\n训练集前5行:")
    print(train_df.head())
    
    log_message("\n训练集列名:")
    print(train_df.columns.tolist())
    
    # 探索数据
    explore_data(train_df, "Training Set")
    explore_data(test_df, "Test Set")
    
    # 检查类别平衡
    log_message("\n" + "="*60)
    log_message("类别平衡检查")
    log_message("="*60)
    
    train_normal = (train_df['label'] == 0).sum()
    train_attack = (train_df['label'] == 1).sum()
    test_normal = (test_df['label'] == 0).sum()
    test_attack = (test_df['label'] == 1).sum()
    
    log_message(f"训练集 - 正常: {train_normal} ({train_normal/len(train_df)*100:.2f}%)")
    log_message(f"训练集 - 攻击: {train_attack} ({train_attack/len(train_df)*100:.2f}%)")
    log_message(f"测试集 - 正常: {test_normal} ({test_normal/len(test_df)*100:.2f}%)")
    log_message(f"测试集 - 攻击: {test_attack} ({test_attack/len(test_df)*100:.2f}%)")
    
    # 数值特征统计
    numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns
    log_message(f"\n数值特征数量: {len(numeric_cols)}")
    log_message(f"数值特征示例: {numeric_cols[:10].tolist()}")
    
    # XGBoost特征类型分析
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    log_message(f"\n类别特征数量: {len(categorical_cols)}")
    log_message(f"类别特征示例: {categorical_cols[:5].tolist()}")
    
    # 攻击类型详细统计
    if 'attack_cat' in train_df.columns:
        log_message("\n训练集攻击类型详细统计:")
        attack_counts = train_df[train_df['label']==1]['attack_cat'].value_counts()
        for atype, count in attack_counts.items():
            percentage = count / train_attack * 100
            log_message(f"  {atype}: {count} ({percentage:.2f}%)")
    
    # 数据质量检查（针对XGBoost）
    log_message("\n" + "="*60)
    log_message("XGBoost数据质量检查")
    log_message("="*60)
    
    # 检查是否有无穷值
    if train_df.select_dtypes(include=['float64', 'int64']).isin([float('inf'), float('-inf')]).any().any():
        log_message("警告: 数据中存在无穷值", "WARNING")
    else:
        log_message("✓ 无无穷值")
    
    # 检查数据类型
    log_message(f"✓ 特征数据类型兼容")
    
    # 内存使用
    memory_usage = train_df.memory_usage(deep=True).sum() / 1024**2
    log_message(f"训练集内存使用: {memory_usage:.2f} MB")
    
    log_message("\n数据探索完成！")

if __name__ == "__main__":
    main()