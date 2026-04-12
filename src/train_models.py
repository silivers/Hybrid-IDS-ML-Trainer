# src/train_models.py
"""
模型训练脚本 - XGBoost入侵检测系统
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import xgboost as xgb
from src.config import XGB_PARAMS
from src.utils import save_model
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
    
    print("\n  🚀 开始训练...")
    train_start = time.time()
    
    model = xgb.XGBClassifier(**params)
    
    try:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    except TypeError:
        model.fit(X_train, y_train, verbose=True)
    
    train_time = time.time() - train_start
    print(f"\n  ✅ 训练完成！耗时: {train_time:.2f}秒")
    
    print("\n  📊 在测试集上预测...")
    predict_start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - predict_start
    
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
    
    metrics = {
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4),
        'TrainTime': round(train_time, 2)
    }
    
    return model, metrics

def train_xgboost_model():
    """训练XGBoost模型"""
    print("\n" + "="*70)
    print("🎯 XGBoost入侵检测系统 - 模型训练")
    print("="*70)
    
    print("\n📂 步骤 1/3: 加载预处理数据...")
    load_start = time.time()
    
    try:
        X_train, X_test, y_train, y_test = get_data()
        print(f"  ✅ 数据加载成功！")
        print(f"     训练集: {X_train.shape}")
        print(f"     测试集: {X_test.shape}")
        print(f"  ⏱️ 加载耗时: {time.time() - load_start:.2f}秒")
    except Exception as e:
        print(f"  ❌ 数据加载失败: {e}")
        return None, None
    
    print("\n📂 步骤 2/3: 训练XGBoost模型...")
    xgb_start = time.time()
    model, metrics = train_xgboost_with_progress(X_train, y_train, X_test, y_test, XGB_PARAMS)
    print(f"\n  ⏱️ XGBoost训练总耗时: {time.time() - xgb_start:.2f}秒")
    
    print("\n📂 步骤 3/3: 保存模型...")
    save_model(model, "xgboost")
    
    print("\n" + "="*70)
    print("✨ XGBoost模型训练完成！")
    print("="*70)
    print(f"\n📊 最终性能:")
    print(f"   准确率: {metrics['Accuracy']:.4f}")
    print(f"   精确率: {metrics['Precision']:.4f}")
    print(f"   召回率: {metrics['Recall']:.4f}")
    print(f"   F1分数: {metrics['F1-Score']:.4f}")
    print(f"\n💾 模型保存位置: models/xgboost.pkl")
    
    return metrics, model

def main():
    print("\n🚀 启动XGBoost模型训练...")
    try:
        metrics, model = train_xgboost_model()
        if model is not None:
            print("\n✅ 训练成功！使用以下命令生成报告:")
            print("   python src/report_generator.py")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()