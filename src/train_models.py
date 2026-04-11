# src/train_models.py
"""
模型训练脚本 - XGBoost入侵检测系统
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import time
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import xgboost as xgb
from src.config import XGB_PARAMS, RANDOM_STATE
from src.utils import save_model, log_message
from src.preprocess import get_data

def train_xgboost_with_progress(X_train, y_train, X_test, y_test, params):
    """训练XGBoost并显示详细进度"""
    print("\n" + "="*60)
    print("⚡ XGBoost模型训练")
    print("="*60)
    
    print(f"  参数配置:")
    for key, value in params.items():
        print(f"    {key}: {value}")
    print(f"\n  训练集大小: {X_train.shape}")
    print(f"  测试集大小: {X_test.shape}")
    
    start_time = time.time()
    
    # 创建模型
    print("\n  📋 创建XGBoost模型实例...")
    model = xgb.XGBClassifier(**params)
    
    # 训练
    print("  🚀 开始训练...")
    print("  💡 XGBoost训练中，请耐心等待...")
    train_start = time.time()
    
    try:
        # 使用验证集显示训练进度
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=True  # 显示每轮进度
        )
    except TypeError:
        # 降级方案
        print("  ⚠️ 使用简化训练模式...")
        model.fit(X_train, y_train, verbose=True)
    
    train_time = time.time() - train_start
    print(f"\n  ✅ 训练完成！")
    print(f"  ⏱️ 训练耗时: {train_time:.2f}秒 ({train_time/60:.2f}分钟)")
    
    # 预测
    print("\n  📊 在测试集上预测...")
    predict_start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - predict_start
    print(f"  ✅ 预测完成！耗时: {predict_time:.2f}秒")
    
    # 评估
    print("\n  📈 计算性能指标...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    total_time = time.time() - start_time
    
    print(f"\n  📊 性能指标:")
    print(f"    准确率 (Accuracy):  {accuracy:.4f}")
    print(f"    精确率 (Precision): {precision:.4f}")
    print(f"    召回率 (Recall):    {recall:.4f}")
    print(f"    F1分数 (F1-Score):  {f1:.4f}")
    print(f"  ⏱️ 总耗时: {total_time:.2f}秒")
    
    # 返回模型和指标
    metrics = {
        'Model': 'XGBoost',
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4),
        'TrainTime': round(train_time, 2),
        'PredictTime': round(predict_time, 2)
    }
    
    return model, metrics

def train_xgboost_model():
    """训练XGBoost模型"""
    print("\n" + "="*70)
    print("🎯 XGBoost入侵检测系统 - 模型训练")
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
        
        # 显示标签分布
        train_attack_rate = np.sum(y_train==1) / len(y_train) * 100
        test_attack_rate = np.sum(y_test==1) / len(y_test) * 100
        print(f"     训练集攻击比例: {train_attack_rate:.2f}%")
        print(f"     测试集攻击比例: {test_attack_rate:.2f}%")
        
    except Exception as e:
        print(f"  ❌ 数据加载失败: {e}")
        print("  提示: 请先运行数据预处理: python src/preprocess.py")
        return None, None
    
    # 训练XGBoost模型
    print("\n📂 步骤 2/3: 训练XGBoost模型...")
    print("="*70)
    
    xgb_start = time.time()
    model, metrics = train_xgboost_with_progress(X_train, y_train, X_test, y_test, XGB_PARAMS)
    print(f"\n  ⏱️ XGBoost训练总耗时: {time.time() - xgb_start:.2f}秒")
    
    # 保存模型
    print("\n📂 步骤 3/3: 保存模型...")
    save_model(model, "xgboost")
    
    # 保存训练结果
    results_df = pd.DataFrame([metrics])
    results_df.to_csv("models/xgboost_training_results.csv", index=False)
    print(f"  💾 训练结果已保存: models/xgboost_training_results.csv")
    
    # 输出最终总结
    print("\n" + "="*70)
    print("✨ XGBoost模型训练完成！")
    print("="*70)
    print(f"\n📊 最终性能:")
    print(f"   准确率:  {metrics['Accuracy']:.4f}")
    print(f"   精确率:  {metrics['Precision']:.4f}")
    print(f"   召回率:  {metrics['Recall']:.4f}")
    print(f"   F1分数:  {metrics['F1-Score']:.4f}")
    print(f"\n⏱️ 训练时间: {metrics['TrainTime']:.2f}秒")
    print(f"💾 模型保存位置: models/xgboost.pkl")
    
    return results_df, model

def main():
    """主函数"""
    print("\n🚀 启动XGBoost模型训练...")
    print("⚠️  注意: XGBoost训练可能需要几分钟时间，请耐心等待\n")
    
    try:
        results_df, model = train_xgboost_model()
        
        if model is not None:
            print("\n✅ 训练成功！可以使用以下命令评估模型:")
            print("   python src/evaluate.py")
            print("\n或使用模型进行预测:")
            print("   from src.utils import load_model")
            print("   model = load_model('xgboost')")
            
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()