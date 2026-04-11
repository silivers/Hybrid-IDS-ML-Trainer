# src/hyperparameter_tuning.py
"""
超参数调优脚本 - XGBoost模型优化
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import xgboost as xgb
from src.config import RANDOM_STATE, MODELS_DIR
from src.utils import log_message, save_model

def get_preprocessed_data():
    """获取预处理后的数据"""
    try:
        from src.preprocess import main as preprocess_main
        result = preprocess_main()
        if result is None:
            return None, None, None, None
        X_train, X_test, y_train, y_test = result
        return X_train, X_test, y_train, y_test
    except Exception as e:
        log_message(f"数据加载失败: {e}", "ERROR")
        return None, None, None, None

def tune_xgboost(X_train, y_train, X_test, y_test):
    """调优XGBoost超参数"""
    print("\n" + "="*60)
    print("⚡ XGBoost超参数调优")
    print("="*60)
    
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.03, 0.05, 0.07],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    total_combinations = (len(param_grid['n_estimators']) * 
                         len(param_grid['max_depth']) * 
                         len(param_grid['learning_rate']) * 
                         len(param_grid['subsample']) *
                         len(param_grid['colsample_bytree']))
    
    print(f"\n📊 总参数组合数: {total_combinations}")
    
    base_model = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    print("\n🚀 开始网格搜索...")
    start_time = time.time()
    
    grid_search = GridSearchCV(
        base_model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"\n✅ 网格搜索完成！耗时: {elapsed:.2f}秒")
    print(f"\n📊 最佳参数: {grid_search.best_params_}")
    print(f"📊 最佳交叉验证F1分数: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    test_f1 = f1_score(y_test, y_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    
    print(f"\n📊 测试集性能:")
    print(f"   准确率: {test_accuracy:.4f}")
    print(f"   精确率: {test_precision:.4f}")
    print(f"   召回率: {test_recall:.4f}")
    print(f"   F1分数: {test_f1:.4f}")
    
    save_model(best_model, "xgboost_tuned")
    
    return best_model, grid_search.best_params_

def main():
    print("\n" + "="*60)
    print("🎯 XGBoost超参数调优工具")
    print("="*60)
    
    print("\n📂 加载预处理数据...")
    X_train, X_test, y_train, y_test = get_preprocessed_data()
    
    if X_train is None:
        print("❌ 数据加载失败，请先运行数据预处理")
        return
    
    print(f"✅ 数据加载成功！训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    confirm = input("\n是否开始调优？(y/n): ").strip().lower()
    if confirm != 'y':
        print("取消调优")
        return
    
    total_start = time.time()
    best_model, best_params = tune_xgboost(X_train, y_train, X_test, y_test)
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*60)
    print(f"🎉 超参数调优完成！耗时: {total_elapsed:.2f}秒")
    print(f"✨ 最佳参数: {best_params}")
    print(f"💾 最佳模型已保存: {MODELS_DIR / 'xgboost_tuned.pkl'}")
    print("="*60)

if __name__ == "__main__":
    main()