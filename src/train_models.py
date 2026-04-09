"""
模型训练脚本 - 训练多个模型并保存
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import xgboost as xgb
from src.config import RF_PARAMS, XGB_PARAMS, RANDOM_STATE
from src.utils import save_model, log_message
from src.preprocess import get_preprocessed_data

def get_preprocessed_data():
    """获取预处理后的数据（如果没有则运行预处理）"""
    try:
        from src.preprocess import main as preprocess_main
        return preprocess_main()
    except:
        from src.preprocess import get_data
        return get_data()

def train_models():
    log_message("="*60)
    log_message("开始模型训练")
    log_message("="*60)
    
    # 获取预处理数据
    log_message("加载预处理数据...")
    
    # 尝试从预处理模块获取数据
    try:
        from src.preprocess import main as preprocess_main
        X_train, X_test, y_train, y_test = preprocess_main()
    except:
        # 直接运行预处理
        from src.preprocess import main
        X_train, X_test, y_train, y_test = main()
    
    log_message(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 定义要训练的模型
    models = {
        'RandomForest': RandomForestClassifier(**RF_PARAMS),
        'XGBoost': xgb.XGBClassifier(**XGB_PARAMS),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    }
    
    results = []
    
    for name, model in models.items():
        log_message(f"\n{'='*50}")
        log_message(f"训练模型: {name}")
        log_message(f"{'='*50}")
        
        # 训练
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        log_message(f"Accuracy:  {accuracy:.4f}")
        log_message(f"Precision: {precision:.4f}")
        log_message(f"Recall:    {recall:.4f}")
        log_message(f"F1-Score:  {f1:.4f}")
        
        # 保存结果
        results.append({
            'Model': name,
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4)
        })
        
        # 保存模型
        save_model(model, name.lower().replace(' ', '_'))
    
    # 输出结果对比
    results_df = pd.DataFrame(results)
    log_message("\n" + "="*60)
    log_message("模型性能对比")
    log_message("="*60)
    print("\n", results_df.to_string(index=False))
    
    # 找出最佳模型
    best_idx = results_df['F1-Score'].idxmax()
    best_model = results_df.loc[best_idx, 'Model']
    best_f1 = results_df.loc[best_idx, 'F1-Score']
    log_message(f"\n最佳模型: {best_model} (F1-Score: {best_f1:.4f})")
    
    return results_df, models

if __name__ == "__main__":
    results_df, models = train_models()