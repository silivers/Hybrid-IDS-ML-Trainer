# src/train_models.py
"""
模型训练脚本 - XGBoost入侵检测系统
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import xgboost as xgb
from src.config import XGB_PARAMS, SELECTED_FEATURES, MODELS_DIR  # 添加 MODELS_DIR 导入
from src.utils import save_model, log_message
from src.preprocess import get_data

def train_xgboost_with_progress(X_train, y_train, X_test, y_test, params, feature_names=None):
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
        # 如果提供了特征名称，设置给模型
        if feature_names is not None and len(feature_names) == X_train.shape[1]:
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
            # 保存特征名称到模型属性
            model._feature_names = feature_names
            log_message(f"已设置 {len(feature_names)} 个特征名称")
        else:
            model.fit(X_train, y_train, verbose=True)
            if feature_names is not None:
                log_message(f"特征名称数量({len(feature_names)})与数据特征数({X_train.shape[1]})不匹配", "WARNING")
            else:
                log_message("未提供特征名称", "WARNING")
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

# src/train_models.py (修复部分)
def train_xgboost_model():
    """训练XGBoost模型"""
    print("\n" + "="*70)
    print("🎯 XGBoost入侵检测系统 - 模型训练")
    print("="*70)
    
    print("\n📂 步骤 1/3: 加载预处理数据...")
    load_start = time.time()
    
    try:
        # 使用修改后的 preprocess_and_cache 获取特征名称
        from src.preprocess import preprocess_and_cache
        result = preprocess_and_cache()
        if result is None:
            print("❌ 数据加载失败")
            return None, None
        
        if len(result) == 5:
            X_train, X_test, y_train, y_test, feature_cols = result
        else:
            X_train, X_test, y_train, y_test = result
            # 尝试从文件读取特征名称
            feature_names_path = MODELS_DIR / "xgboost_feature_names.txt"
            if feature_names_path.exists():
                with open(feature_names_path, 'r', encoding='utf-8') as f:
                    feature_cols = [line.strip() for line in f if line.strip()]
            else:
                from src.config import SELECTED_FEATURES
                feature_cols = SELECTED_FEATURES.copy()
        
        print(f"  ✅ 数据加载成功！")
        print(f"     训练集: {X_train.shape}")
        print(f"     测试集: {X_test.shape}")
        print(f"     特征数量: {X_train.shape[1]}")
        print(f"     特征列表: {feature_cols[:5]}...")
        print(f"  ⏱️ 加载耗时: {time.time() - load_start:.2f}秒")
    except Exception as e:
        print(f"  ❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # 使用实际的特征名称
    feature_names = feature_cols
    
    # 如果特征数量不匹配，输出警告
    if len(feature_names) != X_train.shape[1]:
        log_message(f"⚠️ 警告：特征名称数量({len(feature_names)})与数据特征数({X_train.shape[1]})不匹配", "WARNING")
        log_message(f"使用通用特征名称", "WARNING")
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    
    print("\n📂 步骤 2/3: 训练XGBoost模型...")
    xgb_start = time.time()
    model, metrics = train_xgboost_with_progress(X_train, y_train, X_test, y_test, XGB_PARAMS, feature_names)
    print(f"\n  ⏱️ XGBoost训练总耗时: {time.time() - xgb_start:.2f}秒")
    
    print("\n📂 步骤 3/3: 保存模型...")
    save_model(model, "xgboost")
    
    # 保存特征名称
    try:
        feature_names_path = MODELS_DIR / "xgboost_feature_names.txt"
        with open(feature_names_path, 'w', encoding='utf-8') as f:
            for name in feature_names:
                f.write(name + '\n')
        log_message(f"特征名称已保存: {feature_names_path}")
    except Exception as e:
        log_message(f"保存特征名称失败: {e}", "WARNING")
    
    print("\n" + "="*70)
    print("✨ XGBoost模型训练完成！")
    print("="*70)
    print(f"\n📊 最终性能:")
    print(f"   准确率: {metrics['Accuracy']:.4f}")
    print(f"   精确率: {metrics['Precision']:.4f}")
    print(f"   召回率: {metrics['Recall']:.4f}")
    print(f"   F1分数: {metrics['F1-Score']:.4f}")
    print(f"\n💾 模型保存位置: models/xgboost.pkl")
    print(f"📋 特征数量: {X_train.shape[1]}")
    print(f"📋 特征列表: {feature_names[:10]}...")
    
    return metrics, model

def main():
    print("\n🚀 启动XGBoost模型训练...")
    try:
        metrics, model = train_xgboost_model()
        if model is not None:
            print("\n✅ 训练成功！")

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()