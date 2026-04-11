"""
数据预处理脚本 - XGBoost入侵检测系统
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.config import DATA_DIR, TRAIN_FILE, TEST_FILE, CATEGORICAL_COLS, DROP_COLS
from src.utils import load_data, save_model, log_message

def preprocess_for_xgboost(df, fit_encoders=True, encoders=None, scaler=None):
    """为XGBoost准备数据的预处理函数"""
    log_message("开始XGBoost数据预处理...")
    df_processed = df.copy()
    
    cols_to_drop = [col for col in DROP_COLS if col in df_processed.columns]
    if cols_to_drop:
        df_processed = df_processed.drop(columns=cols_to_drop)
        log_message(f"删除列: {cols_to_drop}")
    
    if df_processed.isnull().sum().sum() > 0:
        df_processed = df_processed.fillna(0)
        log_message("填充缺失值为0")
    
    if encoders is None:
        encoders = {}
    
    for col in CATEGORICAL_COLS:
        if col in df_processed.columns:
            if fit_encoders:
                encoders[col] = LabelEncoder()
                df_processed[col] = encoders[col].fit_transform(df_processed[col].astype(str))
                log_message(f"编码特征: {col}")
            else:
                df_processed[col] = df_processed[col].astype(str)
                known_classes = set(encoders[col].classes_)
                unknown_mask = ~df_processed[col].isin(known_classes)
                if unknown_mask.any():
                    log_message(f"警告: {col}中有{unknown_mask.sum()}个未知类别", "WARNING")
                    df_processed.loc[unknown_mask, col] = -1
                    df_processed.loc[~unknown_mask, col] = encoders[col].transform(df_processed.loc[~unknown_mask, col])
                else:
                    df_processed[col] = encoders[col].transform(df_processed[col])
    
    if 'label' in df_processed.columns:
        y = df_processed['label'].values
        cols_to_drop_for_X = ['label']
        if 'attack_cat' in df_processed.columns:
            cols_to_drop_for_X.append('attack_cat')
        X = df_processed.drop(columns=cols_to_drop_for_X)
    else:
        X = df_processed
        y = None
    
    log_message(f"特征数量: {X.shape[1]}")
    
    if fit_encoders:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        log_message("特征标准化完成")
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, y, encoders, scaler

def main():
    log_message("="*60)
    log_message("XGBoost入侵检测系统 - 数据预处理")
    log_message("="*60)
    
    train_path = DATA_DIR / TRAIN_FILE
    test_path = DATA_DIR / TEST_FILE
    
    if not train_path.exists():
        log_message(f"训练集文件不存在: {train_path}", "ERROR")
        return None
    
    if not test_path.exists():
        log_message(f"测试集文件不存在: {test_path}", "ERROR")
        return None
    
    train_df, test_df = load_data(train_path, test_path)
    
    log_message(f"\n原始训练集形状: {train_df.shape}")
    log_message(f"原始测试集形状: {test_df.shape}")
    
    log_message("\n预处理训练集...")
    X_train, y_train, encoders, scaler = preprocess_for_xgboost(train_df, fit_encoders=True)
    
    log_message("\n预处理测试集...")
    X_test, y_test, _, _ = preprocess_for_xgboost(test_df, fit_encoders=False, 
                                                   encoders=encoders, scaler=scaler)
    
    log_message(f"\n预处理完成！")
    log_message(f"训练集形状: {X_train.shape}")
    log_message(f"测试集形状: {X_test.shape}")
    
    save_model(encoders, "xgboost_label_encoders")
    save_model(scaler, "xgboost_scaler")
    
    log_message("\n数据预处理成功完成！")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    result = main()
    if result:
        X_train, X_test, y_train, y_test = result
        print(f"\n最终数据形状:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")