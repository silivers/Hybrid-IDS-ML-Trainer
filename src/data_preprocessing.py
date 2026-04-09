"""
数据预处理脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.config import DATA_DIR, TRAIN_FILE, TEST_FILE
from src.utils import load_data, preprocess_data, save_model, log_message

def main():
    log_message("="*60)
    log_message("开始数据预处理")
    log_message("="*60)
    
    # 数据文件路径
    train_path = DATA_DIR / TRAIN_FILE
    test_path = DATA_DIR / TEST_FILE
    
    # 加载原始数据
    train_df, test_df = load_data(train_path, test_path)
    
    # 预处理训练集
    log_message("\n预处理训练集...")
    X_train, y_train, encoders, scaler = preprocess_data(train_df, fit_encoders=True)
    
    # 预处理测试集（使用训练集的编码器和标准化器）
    log_message("\n预处理测试集...")
    X_test, y_test, _, _ = preprocess_data(test_df, fit_encoders=False, 
                                           encoders=encoders, scaler=scaler)
    
    # 保存预处理后的数据（可选，为了方便下次使用）
    # 这里不保存到文件，直接在内存中使用
    
    log_message(f"\n预处理完成！")
    log_message(f"训练集形状: {X_train.shape}")
    log_message(f"测试集形状: {X_test.shape}")
    log_message(f"训练集标签分布: 正常={np.sum(y_train==0)}, 攻击={np.sum(y_train==1)}")
    log_message(f"测试集标签分布: 正常={np.sum(y_test==0)}, 攻击={np.sum(y_test==1)}")
    
    # 保存编码器和标准化器
    save_model(encoders, "label_encoders")
    save_model(scaler, "scaler")
    
    # 返回处理后的数据（供其他脚本使用）
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = main()