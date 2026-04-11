"""
工具函数模块 - XGBoost入侵检测系统
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import MODELS_DIR

def log_message(message, level="INFO"):
    """日志打印"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}")

def load_data(train_path, test_path):
    """加载训练集和测试集"""
    log_message(f"加载训练集: {train_path}")
    train_df = pd.read_csv(train_path)
    log_message(f"训练集大小: {train_df.shape}")
    
    log_message(f"加载测试集: {test_path}")
    test_df = pd.read_csv(test_path)
    log_message(f"测试集大小: {test_df.shape}")
    
    return train_df, test_df

def save_model(model, name):
    """保存模型"""
    model_path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, model_path)
    log_message(f"模型已保存: {model_path}")
    return model_path

def load_model(name):
    """加载模型"""
    model_path = MODELS_DIR / f"{name}.pkl"
    if model_path.exists():
        log_message(f"加载模型: {model_path}")
        return joblib.load(model_path)
    else:
        log_message(f"模型不存在: {model_path}", "ERROR")
        return None