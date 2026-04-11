"""
数据预处理模块 - XGBoost入侵检测系统
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from src.config import DATA_DIR, TRAIN_FILE, TEST_FILE, CATEGORICAL_COLS, DROP_COLS
from src.utils import log_message, save_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

_CACHED_DATA = None

def safe_label_encode(train_series, test_series, col_name):
    """安全地处理标签编码"""
    train_str = train_series.astype(str)
    test_str = test_series.astype(str)
    
    le = LabelEncoder()
    train_encoded = le.fit_transform(train_str)
    known_classes = set(le.classes_)
    
    unknown_count = 0
    def encode_test_value(val):
        nonlocal unknown_count
        if val in known_classes:
            return le.transform([val])[0]
        else:
            unknown_count += 1
            return -1
    
    test_encoded = np.array([encode_test_value(val) for val in test_str])
    
    if unknown_count > 0:
        log_message(f"{col_name}中有{unknown_count}个未知类别编码为-1", "WARNING")
    
    return train_encoded, test_encoded, le

def preprocess_for_xgboost(train_df, test_df):
    """为XGBoost优化的数据预处理"""
    log_message("\n开始XGBoost数据预处理...")
    
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    cols_to_drop = [col for col in DROP_COLS if col in train_processed.columns]
    if cols_to_drop:
        train_processed = train_processed.drop(columns=cols_to_drop)
        test_processed = test_processed.drop(columns=cols_to_drop)
        log_message(f"删除列: {cols_to_drop}")
    
    if train_processed.isnull().sum().sum() > 0:
        train_processed = train_processed.fillna(0)
        test_processed = test_processed.fillna(0)
    
    log_message("\n处理类别特征...")
    encoders = {}
    
    for col in CATEGORICAL_COLS:
        if col in train_processed.columns:
            log_message(f"  处理列: {col}")
            train_processed[col], test_processed[col], encoders[col] = safe_label_encode(
                train_processed[col], test_processed[col], col
            )
    
    log_message("\n分离特征和标签...")
    
    y_train = train_processed['label'].values
    cols_to_drop_for_X = ['label']
    if 'attack_cat' in train_processed.columns:
        cols_to_drop_for_X.append('attack_cat')
    X_train = train_processed.drop(columns=cols_to_drop_for_X)
    
    y_test = test_processed['label'].values
    X_test = test_processed.drop(columns=cols_to_drop_for_X)
    
    if list(X_train.columns) != list(X_test.columns):
        common_cols = list(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
        log_message(f"使用共同特征列: {len(common_cols)}")
    
    log_message("\n标准化数值特征...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, encoders, scaler

def preprocess_and_cache():
    """预处理数据并缓存"""
    global _CACHED_DATA
    
    if _CACHED_DATA is not None:
        log_message("使用缓存的数据")
        return _CACHED_DATA
    
    log_message("="*60)
    log_message("XGBoost入侵检测系统 - 数据预处理")
    log_message("="*60)
    
    train_path = DATA_DIR / TRAIN_FILE
    test_path = DATA_DIR / TEST_FILE
    
    if not train_path.exists():
        log_message(f"错误: 训练集文件不存在: {train_path}", "ERROR")
        return None
    if not test_path.exists():
        log_message(f"错误: 测试集文件不存在: {test_path}", "ERROR")
        return None
    
    log_message(f"\n加载训练集: {train_path}")
    train_df = pd.read_csv(train_path)
    log_message(f"训练集大小: {train_df.shape}")
    
    log_message(f"\n加载测试集: {test_path}")
    test_df = pd.read_csv(test_path)
    log_message(f"测试集大小: {test_df.shape}")
    
    X_train, X_test, y_train, y_test, encoders, scaler = preprocess_for_xgboost(train_df, test_df)
    
    save_model(encoders, "xgboost_label_encoders")
    save_model(scaler, "xgboost_scaler")
    
    log_message("\n✅ 预处理完成！")
    
    _CACHED_DATA = (X_train, X_test, y_train, y_test)
    return _CACHED_DATA

def get_data():
    """获取预处理后的数据"""
    result = preprocess_and_cache()
    if result is None:
        raise ValueError("数据预处理失败")
    return result

def main():
    try:
        X_train, X_test, y_train, y_test = preprocess_and_cache()
        if X_train is not None:
            log_message("\n✅ XGBoost数据预处理成功！")
            return X_train, X_test, y_train, y_test
    except Exception as e:
        log_message(f"预处理失败: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
    if result:
        X_train, X_test, y_train, y_test = result
        print(f"\n最终数据形状:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")