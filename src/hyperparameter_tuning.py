"""
超参数调优脚本 - 使用网格搜索优化模型
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import xgboost as xgb
from src.config import RANDOM_STATE
from src.utils import log_message, save_model

def get_preprocessed_data():
    """获取预处理后的数据"""
    try:
        from src.preprocess import main as preprocess_main
        return preprocess_main()
    except:
        log_message("请先运行数据预处理", "ERROR")
        return None, None, None, None

def tune_random_forest(X_train, y_train, X_test, y_test):
    """调优随机森林"""
    log_message("\n" + "="*60)
    log_message("随机森林超参数调优")
    log_message("="*60)
    
    # 参数网格
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 15, 20],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 4, 8]
    }
    
    # 基础模型
    base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    
    # 网格搜索
    log_message("开始网格搜索...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,  # 使用3折交叉验证（数据量不大）
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    log_message(f"\n最佳参数: {grid_search.best_params_}")
    log_message(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")
    
    # 测试集评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred)
    log_message(f"测试集F1分数: {test_f1:.4f}")
    
    # 保存最佳模型
    save_model(best_model, "random_forest_tuned")
    
    return best_model, grid_search.best_params_

def tune_xgboost(X_train, y_train, X_test, y_test):
    """调优XGBoost"""
    log_message("\n" + "="*60)
    log_message("XGBoost超参数调优")
    log_message("="*60)
    
    # 参数网格（简化版，因为全网格会很慢）
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.03, 0.05, 0.07],
        'subsample': [0.7, 0.8, 0.9]
    }
    
    # 基础模型
    base_model = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # 网格搜索
    log_message("开始网格搜索（可能需要几分钟）...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    log_message(f"\n最佳参数: {grid_search.best_params_}")
    log_message(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")
    
    # 测试集评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred)
    log_message(f"测试集F1分数: {test_f1:.4f}")
    
    # 保存最佳模型
    save_model(best_model, "xgboost_tuned")
    
    return best_model, grid_search.best_params_

def main():
    log_message("="*60)
    log_message("超参数调优")
    log_message("="*60)
    
    # 获取数据
    X_train, X_test, y_train, y_test = get_preprocessed_data()
    
    if X_train is None:
        return
    
    log_message(f"训练集: {X_train.shape}")
    
    # 选择要调优的模型（可以单独运行）
    log_message("\n选择要调优的模型:")
    log_message("1. 随机森林")
    log_message("2. XGBoost")
    log_message("3. 两者都调优")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        tune_random_forest(X_train, y_train, X_test, y_test)
    
    if choice in ['2', '3']:
        tune_xgboost(X_train, y_train, X_test, y_test)
    
    log_message("\n超参数调优完成！")

if __name__ == "__main__":
    main()