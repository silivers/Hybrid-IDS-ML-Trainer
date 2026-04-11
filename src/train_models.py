# src/train_models.py
"""
模型训练脚本 - 训练多个模型并保存（带详细进度输出）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import xgboost as xgb
from src.config import RF_PARAMS, XGB_PARAMS, RANDOM_STATE
from src.utils import save_model, log_message
from src.preprocess import get_data

def print_progress(step, total, message, start_time=None):
    """打印进度信息"""
    percent = (step / total) * 100
    bar_length = 30
    filled = int(bar_length * step // total)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    if start_time:
        elapsed = time.time() - start_time
        if step > 0:
            eta = (elapsed / step) * (total - step)
            time_str = f" | 耗时: {elapsed:.1f}s | 剩余: {eta:.1f}s"
        else:
            time_str = ""
    else:
        time_str = ""
    
    print(f"\r  [{bar}] {percent:.1f}% ({step}/{total}) - {message}{time_str}", end='', flush=True)
    
    if step == total:
        print()  # 换行

def train_random_forest_with_progress(X_train, y_train, X_test, y_test, params):
    """训练随机森林并显示详细进度"""
    print("\n" + "="*60)
    print("🌲 训练随机森林模型")
    print("="*60)
    
    print(f"  参数: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")
    print(f"  训练集大小: {X_train.shape}")
    
    start_time = time.time()
    
    # 创建模型
    print("\n  📋 创建模型实例...")
    model = RandomForestClassifier(**params)
    
    # 训练（随机森林训练过程中无法获取中间进度，但可以计时）
    print("  🚀 开始训练...")
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"  ✅ 训练完成！耗时: {train_time:.2f}秒 ({train_time/60:.2f}分钟)")
    
    # 预测
    print("  📊 预测测试集...")
    predict_start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - predict_start
    print(f"  ✅ 预测完成！耗时: {predict_time:.2f}秒")
    
    # 评估
    print("  📈 计算评估指标...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    total_time = time.time() - start_time
    
    print(f"\n  📊 性能指标:")
    print(f"    准确率:  {accuracy:.4f}")
    print(f"    精确率:  {precision:.4f}")
    print(f"    召回率:  {recall:.4f}")
    print(f"    F1分数:  {f1:.4f}")
    print(f"  ⏱️ 总耗时: {total_time:.2f}秒")
    
    return model, {
        'Model': 'RandomForest',
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4),
        'TrainTime': round(train_time, 2),
        'PredictTime': round(predict_time, 2)
    }

# src/train_models.py - 修复 train_xgboost_with_progress 函数

def train_xgboost_with_progress(X_train, y_train, X_test, y_test, params):
    """训练XGBoost并显示详细进度"""
    print("\n" + "="*60)
    print("⚡ 训练XGBoost模型")
    print("="*60)
    
    print(f"  参数: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, learning_rate={params['learning_rate']}")
    print(f"  训练集大小: {X_train.shape}")
    
    start_time = time.time()
    
    # 创建模型
    print("\n  📋 创建模型实例...")
    model = xgb.XGBClassifier(**params)
    
    # 训练
    print("  🚀 开始训练...")
    print("  💡 提示: XGBoost训练中...")
    train_start = time.time()
    
    # 方法1: 使用回调函数显示进度（推荐）
    try:
        # 新版本XGBoost的写法
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=True  # 显示每轮进度
        )
    except TypeError:
        # 如果上面失败，尝试不使用 eval_set
        print("  ⚠️ 使用简化训练模式...")
        model.fit(X_train, y_train, verbose=True)
    
    train_time = time.time() - train_start
    print(f"\n  ✅ 训练完成！")
    print(f"  ⏱️ 训练耗时: {train_time:.2f}秒 ({train_time/60:.2f}分钟)")
    
    # 预测
    print("  📊 预测测试集...")
    predict_start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - predict_start
    print(f"  ✅ 预测完成！耗时: {predict_time:.2f}秒")
    
    # 评估
    print("  📈 计算评估指标...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    total_time = time.time() - start_time
    
    print(f"\n  📊 性能指标:")
    print(f"    准确率:  {accuracy:.4f}")
    print(f"    精确率:  {precision:.4f}")
    print(f"    召回率:  {recall:.4f}")
    print(f"    F1分数:  {f1:.4f}")
    print(f"  ⏱️ 总耗时: {total_time:.2f}秒")
    
    return model, {
        'Model': 'XGBoost',
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4),
        'TrainTime': round(train_time, 2),
        'PredictTime': round(predict_time, 2)
    }
    
    # 评估
    print("  📈 计算评估指标...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    total_time = time.time() - start_time
    
    print(f"\n  📊 性能指标:")
    print(f"    准确率:  {accuracy:.4f}")
    print(f"    精确率:  {precision:.4f}")
    print(f"    召回率:  {recall:.4f}")
    print(f"    F1分数:  {f1:.4f}")
    print(f"  ⏱️ 总耗时: {total_time:.2f}秒")
    
    return model, {
        'Model': 'XGBoost',
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4),
        'TrainTime': round(train_time, 2),
        'PredictTime': round(predict_time, 2)
    }

def train_logistic_regression_with_progress(X_train, y_train, X_test, y_test):
    """训练逻辑回归并显示进度"""
    print("\n" + "="*60)
    print("📈 训练逻辑回归模型")
    print("="*60)
    
    print(f"  训练集大小: {X_train.shape}")
    
    start_time = time.time()
    
    # 创建模型
    print("\n  📋 创建模型实例...")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    
    # 训练
    print("  🚀 开始训练...")
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"  ✅ 训练完成！耗时: {train_time:.2f}秒")
    
    # 预测
    print("  📊 预测测试集...")
    predict_start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - predict_start
    print(f"  ✅ 预测完成！耗时: {predict_time:.2f}秒")
    
    # 评估
    print("  📈 计算评估指标...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    total_time = time.time() - start_time
    
    print(f"\n  📊 性能指标:")
    print(f"    准确率:  {accuracy:.4f}")
    print(f"    精确率:  {precision:.4f}")
    print(f"    召回率:  {recall:.4f}")
    print(f"    F1分数:  {f1:.4f}")
    print(f"  ⏱️ 总耗时: {total_time:.2f}秒")
    
    return model, {
        'Model': 'LogisticRegression',
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4),
        'TrainTime': round(train_time, 2),
        'PredictTime': round(predict_time, 2)
    }

def train_gradient_boosting_with_progress(X_train, y_train, X_test, y_test):
    """训练梯度提升并显示进度"""
    print("\n" + "="*60)
    print("🌿 训练梯度提升模型")
    print("="*60)
    
    print(f"  训练集大小: {X_train.shape}")
    
    start_time = time.time()
    
    # 创建模型
    print("\n  📋 创建模型实例...")
    model = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=1)
    
    # 训练
    print("  🚀 开始训练...")
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"  ✅ 训练完成！耗时: {train_time:.2f}秒 ({train_time/60:.2f}分钟)")
    
    # 预测
    print("  📊 预测测试集...")
    predict_start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - predict_start
    print(f"  ✅ 预测完成！耗时: {predict_time:.2f}秒")
    
    # 评估
    print("  📈 计算评估指标...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    total_time = time.time() - start_time
    
    print(f"\n  📊 性能指标:")
    print(f"    准确率:  {accuracy:.4f}")
    print(f"    精确率:  {precision:.4f}")
    print(f"    召回率:  {recall:.4f}")
    print(f"    F1分数:  {f1:.4f}")
    print(f"  ⏱️ 总耗时: {total_time:.2f}秒")
    
    return model, {
        'Model': 'GradientBoosting',
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4),
        'TrainTime': round(train_time, 2),
        'PredictTime': round(predict_time, 2)
    }

def train_models():
    """训练所有模型"""
    print("\n" + "="*70)
    print("🎯 入侵检测模型训练系统")
    print("="*70)
    
    # 获取预处理数据
    print("\n📂 步骤 1/3: 加载预处理数据...")
    load_start = time.time()
    
    try:
        X_train, X_test, y_train, y_test = get_data()
        print(f"  ✅ 数据加载成功！")
        print(f"     训练集特征: {X_train.shape}")
        print(f"     训练集标签: {y_train.shape}")
        print(f"     测试集特征: {X_test.shape}")
        print(f"     测试集标签: {y_test.shape}")
        print(f"  ⏱️ 加载耗时: {time.time() - load_start:.2f}秒")
    except Exception as e:
        print(f"  ❌ 数据加载失败: {e}")
        print("  提示: 请先运行数据预处理")
        return None, None
    
    # 训练所有模型
    print("\n📂 步骤 2/3: 训练模型...")
    print("="*70)
    
    results = []
    models = {}
    
    # 1. 逻辑回归（最快）
    print("\n[1/4] 训练逻辑回归...")
    lr_start = time.time()
    model, result = train_logistic_regression_with_progress(X_train, y_train, X_test, y_test)
    results.append(result)
    models['logistic_regression'] = model
    print(f"  ⏱️ 逻辑回归总耗时: {time.time() - lr_start:.2f}秒")
    save_model(model, "logistic_regression")
    
    # 2. 随机森林
    print("\n[2/4] 训练随机森林...")
    rf_start = time.time()
    model, result = train_random_forest_with_progress(X_train, y_train, X_test, y_test, RF_PARAMS)
    results.append(result)
    models['random_forest'] = model
    print(f"  ⏱️ 随机森林总耗时: {time.time() - rf_start:.2f}秒")
    save_model(model, "random_forest")
    
    # 3. 梯度提升
    print("\n[3/4] 训练梯度提升...")
    gb_start = time.time()
    model, result = train_gradient_boosting_with_progress(X_train, y_train, X_test, y_test)
    results.append(result)
    models['gradient_boosting'] = model
    print(f"  ⏱️ 梯度提升总耗时: {time.time() - gb_start:.2f}秒")
    save_model(model, "gradient_boosting")
    
    # 4. XGBoost（最慢，可能耗时最长）
    print("\n[4/4] 训练XGBoost...")
    xgb_start = time.time()
    model, result = train_xgboost_with_progress(X_train, y_train, X_test, y_test, XGB_PARAMS)
    results.append(result)
    models['xgboost'] = model
    print(f"  ⏱️ XGBoost总耗时: {time.time() - xgb_start:.2f}秒")
    save_model(model, "xgboost")
    
    # 结果对比
    print("\n" + "="*70)
    print("📂 步骤 3/3: 生成性能报告")
    print("="*70)
    
    results_df = pd.DataFrame(results)
    
    # 美化输出
    print("\n🏆 模型性能对比表:")
    print("-" * 80)
    print(f"{'模型':<20} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'训练时间':<10}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<20} {row['Accuracy']:<10.4f} {row['Precision']:<10.4f} "
              f"{row['Recall']:<10.4f} {row['F1-Score']:<10.4f} {row['TrainTime']:<10.1f}s")
    
    print("-" * 80)
    
    # 找出最佳模型
    best_idx = results_df['F1-Score'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_f1 = results_df.loc[best_idx, 'F1-Score']
    best_train_time = results_df.loc[best_idx, 'TrainTime']
    
    print(f"\n✨ 最佳模型: {best_model_name}")
    print(f"   F1分数: {best_f1:.4f}")
    print(f"   训练时间: {best_train_time:.1f}秒")
    
    # 显示训练时间对比
    print("\n⏱️ 训练时间对比:")
    print(f"  最快: {results_df.loc[results_df['TrainTime'].idxmin(), 'Model']} ({results_df['TrainTime'].min():.1f}秒)")
    print(f"  最慢: {results_df.loc[results_df['TrainTime'].idxmax(), 'Model']} ({results_df['TrainTime'].max():.1f}秒)")
    
    # 保存结果
    results_df.to_csv("models/training_results.csv", index=False)
    print(f"\n💾 结果已保存到: models/training_results.csv")
    
    total_time = sum(results_df['TrainTime']) + sum(results_df['PredictTime'])
    print(f"\n🎉 所有模型训练完成！总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    
    return results_df, models

if __name__ == "__main__":
    print("\n🚀 启动模型训练...")
    print("⚠️  注意: XGBoost和随机森林训练可能需要几分钟时间\n")
    results_df, models = train_models()