"""
工具函数模块
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import MODELS_DIR, FIGURES_DIR, LOGS_DIR, REPORTS_DIR

# 简单的日志函数
def log_message(message, level="INFO"):
    """简单的日志打印"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}")

def load_data(train_path, test_path):
    """
    加载训练集和测试集
    """
    log_message(f"加载训练集: {train_path}")
    train_df = pd.read_csv(train_path)
    log_message(f"训练集大小: {train_df.shape}")
    
    log_message(f"加载测试集: {test_path}")
    test_df = pd.read_csv(test_path)
    log_message(f"测试集大小: {test_df.shape}")
    
    return train_df, test_df

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

def preprocess_data(df, fit_encoders=True, encoders=None, scaler=None):
    """
    数据预处理
    """
    from src.config import CATEGORICAL_COLS, DROP_COLS
    
    log_message("开始数据预处理...")
    df_processed = df.copy()
    
    # 删除不必要的列
    cols_to_drop = [col for col in DROP_COLS if col in df_processed.columns]
    if cols_to_drop:
        df_processed = df_processed.drop(columns=cols_to_drop)
        log_message(f"删除列: {cols_to_drop}")
    
    # 处理缺失值
    if df_processed.isnull().sum().sum() > 0:
        df_processed = df_processed.fillna(0)
        log_message("填充缺失值为0")
    
    # 处理类别特征
    if encoders is None:
        encoders = {}
    
    for col in CATEGORICAL_COLS:
        if col in df_processed.columns:
            if fit_encoders:
                encoders[col] = LabelEncoder()
                df_processed[col] = encoders[col].fit_transform(df_processed[col].astype(str))
                log_message(f"编码特征: {col}")
            else:
                df_processed[col] = encoders[col].transform(df_processed[col].astype(str))
    
    # 分离特征和标签
    if 'label' in df_processed.columns:
        y = df_processed['label'].values
        # 删除标签列和攻击类别列
        cols_to_drop_for_X = ['label']
        if 'attack_cat' in df_processed.columns:
            cols_to_drop_for_X.append('attack_cat')
        X = df_processed.drop(columns=cols_to_drop_for_X)
    else:
        X = df_processed
        y = None
    
    log_message(f"特征数量: {X.shape[1]}")
    
    # 标准化
    if fit_encoders:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        log_message("特征标准化完成")
    else:
        X_scaled = scaler.transform(X)
        log_message("使用已有的标准化器")
    
    return X_scaled, y, encoders, scaler

def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """绘制混淆矩阵 - 已禁用"""
    # 计算混淆矩阵并打印
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    log_message(f"\n混淆矩阵详情:")
    log_message(f"  True Negatives (正常→正常): {tn}")
    log_message(f"  False Positives (正常→攻击): {fp}")
    log_message(f"  False Negatives (攻击→正常): {fn}")
    log_message(f"  True Positives (攻击→攻击): {tp}")
    
    # 计算指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    log_message(f"\n基于混淆矩阵的指标:")
    log_message(f"  准确率: {accuracy:.4f}")
    log_message(f"  精确率: {precision:.4f}")
    log_message(f"  召回率: {recall:.4f}")
    log_message(f"  F1分数: {f1:.4f}")
    
    if save_path:
        log_message(f"混淆矩阵已保存到: {save_path} (仅数值)")
    
    return cm

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """绘制特征重要性 - 已禁用，只打印"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        log_message("模型没有feature_importances_属性", "WARNING")
        return
    
    # 确保特征名称数量匹配
    if len(feature_names) != len(importances):
        log_message(f"特征名称数量({len(feature_names)})与重要性数量({len(importances)})不匹配", "WARNING")
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    # 排序
    indices = np.argsort(importances)[::-1][:top_n]
    
    # 打印特征重要性
    log_message(f"\nTop {top_n} 重要特征:")
    for i in range(min(top_n, len(indices))):
        log_message(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    if save_path:
        log_message(f"特征重要性已保存到: {save_path} (仅文本)")

def plot_roc_curve(model, X_test, y_test, model_name, save_path=None):
    """绘制ROC曲线 - 已禁用，只打印AUC"""
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        log_message(f"ROC AUC Score: {roc_auc:.4f}")
        
        if save_path:
            log_message(f"ROC曲线数据已保存到: {save_path} (仅数值)")
        
        return roc_auc
    else:
        log_message("模型不支持predict_proba", "WARNING")
        return None

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

# 导入numpy
import numpy as np