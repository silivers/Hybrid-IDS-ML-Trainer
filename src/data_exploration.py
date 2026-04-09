"""
数据探索脚本 - 运行此脚本了解数据特征
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.config import DATA_DIR, TRAIN_FILE, TEST_FILE, FIGURES_DIR
from src.utils import load_data, explore_data, plot_attack_distribution, log_message

def main():
    log_message("="*60)
    log_message("开始数据探索")
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
    
    # 绘制攻击分布图
    plot_attack_distribution(train_df, FIGURES_DIR / "attack_distribution.png")
    
    # 数值特征统计
    numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns
    log_message(f"\n数值特征数量: {len(numeric_cols)}")
    log_message(f"数值特征示例: {numeric_cols[:10].tolist()}")
    
    log_message("\n数据探索完成！")

if __name__ == "__main__":
    main()