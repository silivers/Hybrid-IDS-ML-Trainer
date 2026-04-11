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
    """数据探索函数"""
    log_message(f"\n{'='*60}")
    log_message(f"{name} 探索性分析")
    log_message(f"{'='*60}")
    
    log_message(f"形状: {df.shape}")
    log_message(f"列数: {len(df.columns)}")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        log_message(f"缺失值数量: {missing.sum()}")
    else:
        log_message("无缺失值")
    
    if 'label' in df.columns:
        normal_count = (df['label'] == 0).sum()
        attack_count = (df['label'] == 1).sum()
        log_message(f"\n标签分布:")
        log_message(f"  正常 (0): {normal_count} ({normal_count/len(df)*100:.2f}%)")
        log_message(f"  攻击 (1): {attack_count} ({attack_count/len(df)*100:.2f}%)")
    
    if 'attack_cat' in df.columns:
        log_message(f"\n攻击类型分布:")
        attack_types = df[df['label']==1]['attack_cat'].value_counts()
        for atype, count in attack_types.items():
            log_message(f"  {atype}: {count}")

def main():
    log_message("="*60)
    log_message("XGBoost入侵检测系统 - 数据探索")
    log_message("="*60)
    
    train_path = DATA_DIR / TRAIN_FILE
    test_path = DATA_DIR / TEST_FILE
    
    if not train_path.exists():
        log_message(f"训练集文件不存在: {train_path}", "ERROR")
        return
    
    if not test_path.exists():
        log_message(f"测试集文件不存在: {test_path}", "ERROR")
        return
    
    train_df, test_df = load_data(train_path, test_path)
    
    log_message("\n训练集前5行:")
    print(train_df.head())
    
    explore_data(train_df, "Training Set")
    explore_data(test_df, "Test Set")
    
    log_message("\n数据探索完成！")

if __name__ == "__main__":
    main()