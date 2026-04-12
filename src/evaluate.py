"""
模型评估脚本 - XGBoost入侵检测系统评估（精简版）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc)
from src.utils import load_model, log_message
from src.config import SELECTED_FEATURES

def get_feature_names():
    """获取XGBoost模型的特征名称（精简版）"""
    return SELECTED_FEATURES.copy()

def evaluate_xgboost_model():
    """评估XGBoost模型，返回评估结果字典"""
    log_message("="*60)
    log_message("XGBoost入侵检测系统 - 模型评估（精简版）")
    log_message("="*60)
    
    # 获取预处理数据
    log_message("\n步骤 1/4: 加载预处理数据...")
    try:
        from src.preprocess import main as preprocess_main
        result = preprocess_main()
        if result is None:
            log_message("数据预处理失败", "ERROR")
            return None
        X_train, X_test, y_train, y_test = result
        log_message(f"✓ 数据加载成功，测试集大小: {X_test.shape}")
        log_message(f"  特征数量: {X_test.shape[1]}")
    except Exception as e:
        log_message(f"数据加载失败: {e}", "ERROR")
        return None
    
    # 加载XGBoost模型
    log_message("\n步骤 2/4: 加载XGBoost模型...")
    xgboost_model = load_model("xgboost")
    if xgboost_model is None:
        log_message("XGBoost模型加载失败", "ERROR")
        return None
    log_message("✓ XGBoost模型加载成功")
    
    # 预测
    log_message("\n步骤 3/4: 在测试集上预测...")
    y_pred = xgboost_model.predict(X_test)
    log_message(f"✓ 预测完成，共 {len(y_pred)} 个样本")
    
    # 计算评估指标
    log_message("\n步骤 4/4: 计算评估指标...")
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # 计算AUC
    if hasattr(xgboost_model, 'predict_proba'):
        y_score = xgboost_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = None
        fpr, tpr = None, None
    
    # 特征重要性
    if hasattr(xgboost_model, 'feature_importances_'):
        importances = xgboost_model.feature_importances_
        feature_names = get_feature_names()
        # 确保特征名称数量匹配
        if len(feature_names) != len(importances):
            log_message(f"警告：特征名称数量({len(feature_names)})与重要性数量({len(importances)})不匹配", "WARNING")
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        indices = np.argsort(importances)[::-1]
    else:
        importances = None
        indices = None
        feature_names = None
    
    # 构建评估结果字典
    evaluation_results = {
        'model': xgboost_model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': roc_auc
        },
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        },
        'feature_importance': {
            'importances': importances,
            'indices': indices,
            'feature_names': feature_names
        },
        'roc_curve': {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }
    }
    
    log_message("\n✅ XGBoost模型评估完成！")
    return evaluation_results

def main():
    """主函数"""
    try:
        results = evaluate_xgboost_model()
        if results:
            print("\n💡 使用 report_generator.py 生成详细报告:")
            print("   python src/report_generator.py")
            return results
    except Exception as e:
        log_message(f"评估过程出错: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()