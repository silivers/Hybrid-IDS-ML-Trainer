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

# 全局变量
_CACHED_DATA = None

def inspect_data(df, name="Dataset"):
    """检查数据格式（针对XGBoost优化）"""
    log_message(f"\n{name} 数据信息:")
    log_message(f"  形状: {df.shape}")
    log_message(f"  列数: {len(df.columns)}")
    log_message(f"  列名: {df.columns.tolist()[:10]}...")
    log_message(f"  数据类型:\n{df.dtypes.value_counts()}")
    
    # 检查类别列的值（XGBoost需要编码的列）
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            unique_vals = df[col].unique()[:10]
            log_message(f"  {col} 唯一值示例: {unique_vals}")
    
    return df

def safe_label_encode(train_series, test_series, col_name):
    """
    安全地处理标签编码（处理测试集中的未知类别）
    """
    # 转换为字符串
    train_str = train_series.astype(str)
    test_str = test_series.astype(str)
    
    # 创建编码器
    le = LabelEncoder()
    
    # 在训练集上拟合
    train_encoded = le.fit_transform(train_str)
    
    # 获取已知类别
    known_classes = set(le.classes_)
    
    # 处理测试集中未知的类别
    unknown_count = 0
    def encode_test_value(val):
        nonlocal unknown_count
        if val in known_classes:
            return le.transform([val])[0]
        else:
            unknown_count += 1
            return -1  # 未知类别编码为-1
    
    test_encoded = np.array([encode_test_value(val) for val in test_str])
    
    if unknown_count > 0:
        log_message(f"  列 '{col_name}' 中有 {unknown_count} 个未知类别被编码为 -1", "WARNING")
    
    return train_encoded, test_encoded, le

def preprocess_for_xgboost(train_df, test_df):
    """
    为XGBoost优化的数据预处理
    """
    log_message("\n开始XGBoost数据预处理...")
    
    # 复制数据
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    # 1. 删除不必要的列
    cols_to_drop = [col for col in DROP_COLS if col in train_processed.columns]
    if cols_to_drop:
        train_processed = train_processed.drop(columns=cols_to_drop)
        test_processed = test_processed.drop(columns=cols_to_drop)
        log_message(f"删除列: {cols_to_drop}")
    
    # 2. 处理缺失值（XGBoost可以处理缺失值，但填充0更稳定）
    if train_processed.isnull().sum().sum() > 0:
        log_message("处理缺失值（填充0）...")
        train_processed = train_processed.fillna(0)
        test_processed = test_processed.fillna(0)
    
    # 3. 处理类别特征（XGBoost需要数值输入）
    log_message("\n处理类别特征...")
    encoders = {}
    
    for col in CATEGORICAL_COLS:
        if col in train_processed.columns:
            log_message(f"  处理列: {col}")
            train_processed[col], test_processed[col], encoders[col] = safe_label_encode(
                train_processed[col], test_processed[col], col
            )
            log_message(f"    训练集类别数: {len(encoders[col].classes_)}")
            log_message(f"    训练集值范围: {train_processed[col].min()} ~ {train_processed[col].max()}")
    
    # 4. 分离特征和标签
    log_message("\n分离特征和标签...")
    
    # 训练集
    y_train = train_processed['label'].values
    cols_to_drop_for_X = ['label']
    if 'attack_cat' in train_processed.columns:
        cols_to_drop_for_X.append('attack_cat')  # attack_cat不用于训练
    X_train = train_processed.drop(columns=cols_to_drop_for_X)
    
    # 测试集
    y_test = test_processed['label'].values
    X_test = test_processed.drop(columns=cols_to_drop_for_X)
    
    log_message(f"训练集特征数量: {X_train.shape[1]}")
    log_message(f"测试集特征数量: {X_test.shape[1]}")
    
    # 确保特征列一致
    if list(X_train.columns) != list(X_test.columns):
        log_message("警告: 训练集和测试集特征列不一致!", "WARNING")
        # 找到共同的列
        common_cols = list(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
        log_message(f"使用共同特征列: {len(common_cols)}")
    
    # 5. 标准化数值特征（XGBoost不强制要求，但有助于收敛）
    log_message("\n标准化数值特征...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    log_message(f"标准化完成，特征维度: {X_train_scaled.shape[1]}")
    
    # 6. 数据验证（针对XGBoost）
    log_message("\n数据验证:")
    log_message(f"  训练集 - 正常: {np.sum(y_train==0)}, 攻击: {np.sum(y_train==1)}")
    log_message(f"  测试集 - 正常: {np.sum(y_test==0)}, 攻击: {np.sum(y_test==1)}")
    log_message(f"  训练集特征范围: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
    log_message(f"  测试集特征范围: [{X_test_scaled.min():.2f}, {X_test_scaled.max():.2f}]")
    
    # 检查是否有无穷值
    if np.isinf(X_train_scaled).any() or np.isinf(X_test_scaled).any():
        log_message("警告: 数据中存在无穷值！", "ERROR")
    
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
    
    # 数据文件路径
    train_path = DATA_DIR / TRAIN_FILE
    test_path = DATA_DIR / TEST_FILE
    
    # 检查文件是否存在
    if not train_path.exists():
        log_message(f"错误: 训练集文件不存在: {train_path}", "ERROR")
        return None
    if not test_path.exists():
        log_message(f"错误: 测试集文件不存在: {test_path}", "ERROR")
        return None
    
    # 加载原始数据
    log_message(f"\n加载训练集: {train_path}")
    train_df = pd.read_csv(train_path)
    log_message(f"训练集大小: {train_df.shape}")
    
    log_message(f"\n加载测试集: {test_path}")
    test_df = pd.read_csv(test_path)
    log_message(f"测试集大小: {test_df.shape}")
    
    # 检查数据
    inspect_data(train_df, "训练集")
    inspect_data(test_df, "测试集")
    
    # XGBoost预处理
    X_train, X_test, y_train, y_test, encoders, scaler = preprocess_for_xgboost(train_df, test_df)
    
    # 保存编码器和标准化器（供预测时使用）
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
    """主函数"""
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
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test: {y_test.shape}")
        
        # 显示数据统计信息
        print(f"\n数据统计:")
        print(f"  训练集特征范围: [{X_train.min():.4f}, {X_train.max():.4f}]")
        print(f"  测试集特征范围: [{X_test.min():.4f}, {X_test.max():.4f}]")
        print(f"  训练集攻击比例: {np.sum(y_train==1)/len(y_train)*100:.2f}%")
        print(f"  测试集攻击比例: {np.sum(y_test==1)/len(y_test)*100:.2f}%")