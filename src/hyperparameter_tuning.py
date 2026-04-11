# src/hyperparameter_tuning.py
"""
超参数调优脚本 - XGBoost模型优化
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import time
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import xgboost as xgb
from src.config import RANDOM_STATE, MODELS_DIR, REPORTS_DIR
from src.utils import log_message, save_model

def get_preprocessed_data():
    """获取预处理后的数据"""
    try:
        from src.preprocess import main as preprocess_main
        result = preprocess_main()
        if result is None:
            log_message("数据预处理失败", "ERROR")
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
    
    # 参数网格
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
        eval_metric='logloss',
        n_jobs=-1
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
        verbose=1  # 显示进度
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
    
    # 计算各项指标
    test_f1 = f1_score(y_test, y_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    
    print(f"\n📊 测试集性能:")
    print(f"   准确率: {test_accuracy:.4f}")
    print(f"   精确率: {test_precision:.4f}")
    print(f"   召回率: {test_recall:.4f}")
    print(f"   F1分数: {test_f1:.4f}")
    
    # 显示所有结果排序
    print("\n📊 Top 10 参数组合:")
    results = zip(grid_search.cv_results_['params'], 
                  grid_search.cv_results_['mean_test_score'],
                  grid_search.cv_results_['std_test_score'])
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:10]
    
    for i, (params, score, std) in enumerate(sorted_results, 1):
        print(f"   {i}. F1={score:.4f}±{std:.4f} | {params}")
    
    # 保存调优结果
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(REPORTS_DIR / "xgboost_tuning_results.csv", index=False)
    print(f"\n💾 调优结果已保存: {REPORTS_DIR / 'xgboost_tuning_results.csv'}")
    
    # 保存最佳模型
    save_model(best_model, "xgboost_tuned")
    
    # 对比默认参数和调优后参数
    print("\n📊 参数对比:")
    print(f"   默认参数: {xgboost_default_params()}")
    print(f"   调优参数: {grid_search.best_params_}")
    
    # 性能提升
    from src.train_models import train_xgboost_with_progress
    from src.config import XGB_PARAMS
    
    # 训练默认模型进行对比
    print("\n📊 训练默认模型进行对比...")
    default_model = xgb.XGBClassifier(**XGB_PARAMS)
    default_model.fit(X_train, y_train)
    default_pred = default_model.predict(X_test)
    default_f1 = f1_score(y_test, default_pred)
    
    improvement = (test_f1 - default_f1) / default_f1 * 100
    print(f"\n📈 性能提升:")
    print(f"   默认模型F1: {default_f1:.4f}")
    print(f"   调优模型F1: {test_f1:.4f}")
    print(f"   提升幅度: {improvement:+.2f}%")
    
    return best_model, grid_search.best_params_

def xgboost_default_params():
    """返回XGBoost默认参数"""
    return {
        'n_estimators': 200,
        'max_depth': 10,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

def main():
    print("\n" + "="*60)
    print("🎯 XGBoost超参数调优工具")
    print("="*60)
    
    # 获取数据
    print("\n📂 加载预处理数据...")
    load_start = time.time()
    X_train, X_test, y_train, y_test = get_preprocessed_data()
    
    if X_train is None:
        print("❌ 数据加载失败，请先运行数据预处理")
        print("提示: python src/preprocess.py")
        return
    
    print(f"✅ 数据加载成功！")
    print(f"   训练集: {X_train.shape}")
    print(f"   测试集: {X_test.shape}")
    print(f"   训练集标签: 正常={np.sum(y_train==0)}, 攻击={np.sum(y_train==1)}")
    print(f"   测试集标签: 正常={np.sum(y_test==0)}, 攻击={np.sum(y_test==1)}")
    print(f"⏱️ 加载耗时: {time.time() - load_start:.2f}秒")
    
    # 确认开始调优
    print("\n" + "="*60)
    print("⚠️  注意事项:")
    print("   1. 超参数调优可能需要15-30分钟")
    print("   2. 会使用所有CPU核心进行并行计算")
    print("   3. 调优后的模型将保存为 'xgboost_tuned.pkl'")
    print("="*60)
    
    confirm = input("\n是否开始调优？(y/n): ").strip().lower()
    if confirm != 'y':
        print("取消调优")
        return
    
    # 开始调优
    total_start = time.time()
    best_model, best_params = tune_xgboost(X_train, y_train, X_test, y_test)
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*60)
    print(f"🎉 XGBoost超参数调优完成！")
    print(f"⏱️ 总耗时: {total_elapsed:.2f}秒 ({total_elapsed/60:.2f}分钟)")
    print(f"✨ 最佳参数: {best_params}")
    print(f"💾 最佳模型已保存: {MODELS_DIR / 'xgboost_tuned.pkl'}")
    print("="*60)
    
    # 可选：替换默认模型
    replace = input("\n是否用调优后的模型替换默认XGBoost模型？(y/n): ").strip().lower()
    if replace == 'y':
        import shutil
        shutil.copy(MODELS_DIR / "xgboost_tuned.pkl", MODELS_DIR / "xgboost.pkl")
        print("✅ 已替换默认XGBoost模型")
    
    print("\n💡 提示: 使用调优后的模型进行预测:")
    print("   from src.utils import load_model")
    print("   model = load_model('xgboost_tuned')")

if __name__ == "__main__":
    main()