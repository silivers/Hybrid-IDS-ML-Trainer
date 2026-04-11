# src/hyperparameter_tuning.py (修改关键部分)
"""
超参数调优脚本 - 使用网格搜索优化模型（带进度显示）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import xgboost as xgb
from src.config import RANDOM_STATE
from src.utils import log_message, save_model

def get_preprocessed_data():
    """获取预处理后的数据"""
    try:
        from src.preprocess import get_data
        return get_data()
    except:
        log_message("请先运行数据预处理", "ERROR")
        return None, None, None, None

def tune_random_forest(X_train, y_train, X_test, y_test):
    """调优随机森林（带进度显示）"""
    print("\n" + "="*60)
    print("🌲 随机森林超参数调优")
    print("="*60)
    
    # 参数网格
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 15, 20],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 4, 8]
    }
    
    total_combinations = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * \
                         len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])
    
    print(f"\n📊 参数网格信息:")
    print(f"   总参数组合数: {total_combinations}")
    print(f"   交叉验证折数: 3")
    print(f"   总训练次数: {total_combinations * 3}")
    print(f"\n参数范围:")
    for param, values in param_grid.items():
        print(f"   {param}: {values}")
    
    # 基础模型
    base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    
    # 网格搜索
    print("\n🚀 开始网格搜索...")
    print("💡 提示: 这可能需要5-10分钟，请耐心等待...")
    
    start_time = time.time()
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=2  # 设置为2会显示详细进度
    )
    
    grid_search.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ 网格搜索完成！")
    print(f"⏱️ 耗时: {elapsed:.2f}秒 ({elapsed/60:.2f}分钟)")
    print(f"\n📊 最佳参数: {grid_search.best_params_}")
    print(f"📊 最佳交叉验证F1分数: {grid_search.best_score_:.4f}")
    
    # 测试集评估
    print("\n📈 测试集评估中...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred)
    print(f"📊 测试集F1分数: {test_f1:.4f}")
    
    # 显示所有结果排序
    print("\n📊 Top 5 参数组合:")
    results = zip(grid_search.cv_results_['params'], 
                  grid_search.cv_results_['mean_test_score'],
                  grid_search.cv_results_['std_test_score'])
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:5]
    
    for i, (params, score, std) in enumerate(sorted_results, 1):
        print(f"   {i}. F1={score:.4f}±{std:.4f} | {params}")
    
    # 保存最佳模型
    save_model(best_model, "random_forest_tuned")
    
    return best_model, grid_search.best_params_

def tune_xgboost(X_train, y_train, X_test, y_test):
    """调优XGBoost（带进度显示）"""
    print("\n" + "="*60)
    print("⚡ XGBoost超参数调优")
    print("="*60)
    
    # 参数网格
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.03, 0.05, 0.07],
        'subsample': [0.7, 0.8, 0.9]
    }
    
    total_combinations = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * \
                         len(param_grid['learning_rate']) * len(param_grid['subsample'])
    
    print(f"\n📊 参数网格信息:")
    print(f"   总参数组合数: {total_combinations}")
    print(f"   交叉验证折数: 3")
    print(f"   总训练次数: {total_combinations * 3}")
    print(f"\n参数范围:")
    for param, values in param_grid.items():
        print(f"   {param}: {values}")
    
    # 基础模型
    base_model = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # 网格搜索
    print("\n🚀 开始网格搜索...")
    print("💡 提示: 这可能需要15-30分钟，请耐心等待...")
    
    start_time = time.time()
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=2  # 显示详细进度
    )
    
    grid_search.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ 网格搜索完成！")
    print(f"⏱️ 耗时: {elapsed:.2f}秒 ({elapsed/60:.2f}分钟)")
    print(f"\n📊 最佳参数: {grid_search.best_params_}")
    print(f"📊 最佳交叉验证F1分数: {grid_search.best_score_:.4f}")
    
    # 测试集评估
    print("\n📈 测试集评估中...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred)
    print(f"📊 测试集F1分数: {test_f1:.4f}")
    
    # 显示所有结果排序
    print("\n📊 Top 5 参数组合:")
    results = zip(grid_search.cv_results_['params'], 
                  grid_search.cv_results_['mean_test_score'],
                  grid_search.cv_results_['std_test_score'])
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:5]
    
    for i, (params, score, std) in enumerate(sorted_results, 1):
        print(f"   {i}. F1={score:.4f}±{std:.4f} | {params}")
    
    # 保存最佳模型
    save_model(best_model, "xgboost_tuned")
    
    return best_model, grid_search.best_params_

def main():
    print("\n" + "="*60)
    print("🎯 超参数调优工具")
    print("="*60)
    
    # 获取数据
    print("\n📂 加载预处理数据...")
    load_start = time.time()
    result = get_preprocessed_data()
    
    if result is None:
        return
    
    X_train, X_test, y_train, y_test = result
    print(f"✅ 数据加载成功！")
    print(f"   训练集: {X_train.shape}")
    print(f"   测试集: {X_test.shape}")
    print(f"⏱️ 加载耗时: {time.time() - load_start:.2f}秒")
    
    # 选择要调优的模型
    print("\n" + "="*60)
    print("选择要调优的模型:")
    print("  1. 随机森林 (推荐，较快)")
    print("  2. XGBoost (推荐，较慢但效果可能更好)")
    print("  3. 两者都调优 (耗时最长)")
    print("="*60)
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    total_start = time.time()
    
    if choice in ['1', '3']:
        tune_random_forest(X_train, y_train, X_test, y_test)
    
    if choice in ['2', '3']:
        tune_xgboost(X_train, y_train, X_test, y_test)
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*60)
    print(f"🎉 超参数调优完成！")
    print(f"⏱️ 总耗时: {total_elapsed:.2f}秒 ({total_elapsed/60:.2f}分钟)")
    print("="*60)

if __name__ == "__main__":
    main()