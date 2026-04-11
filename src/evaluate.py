"""
模型评估脚本 - XGBoost入侵检测系统评估
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, accuracy_score, 
                           precision_score, recall_score, f1_score,
                           confusion_matrix, roc_curve, auc)
from src.config import REPORTS_DIR
from src.utils import load_model, log_message

def get_feature_names():
    """获取XGBoost模型的特征名称"""
    # UNSW-NB15的核心特征（43个特征，移除了id, attack_cat, label）
    feature_names = [
        'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 
        'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 
        'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 
        'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 
        'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 
        'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 
        'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
        'ct_dst_src_ltm'
    ]
    return feature_names

def calculate_confusion_matrix_metrics(y_true, y_pred):
    """计算混淆矩阵相关指标"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    log_message(f"\n混淆矩阵详情:")
    log_message(f"  True Negatives (正常→正常): {tn}")
    log_message(f"  False Positives (正常→攻击): {fp}")
    log_message(f"  False Negatives (攻击→正常): {fn}")
    log_message(f"  True Positives (攻击→攻击): {tp}")
    
    # 计算衍生指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'accuracy': accuracy, 'precision': precision, 
        'recall': recall, 'f1': f1
    }

def calculate_roc_auc(model, X_test, y_test):
    """计算ROC曲线和AUC值"""
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        return roc_auc
    else:
        log_message("模型不支持predict_proba", "WARNING")
        return None

def print_feature_importance(model, feature_names, top_n=20):
    """打印特征重要性"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # 确保特征名称数量匹配
        if len(feature_names) != len(importances):
            log_message(f"特征名称数量({len(feature_names)})与重要性数量({len(importances)})不匹配", "WARNING")
            log_message(f"使用默认特征名称 (feature_0 到 feature_{len(importances)-1})")
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # 排序
        indices = np.argsort(importances)[::-1][:top_n]
        
        # 打印特征重要性
        log_message(f"\nTop {top_n} 重要特征:")
        for i in range(min(top_n, len(indices))):
            log_message(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        return importances, indices
    else:
        log_message("模型没有feature_importances_属性", "WARNING")
        return None, None

def evaluate_xgboost_model():
    """评估XGBoost模型"""
    log_message("="*60)
    log_message("XGBoost入侵检测系统 - 模型评估")
    log_message("="*60)
    
    # 获取预处理数据
    log_message("\n步骤 1/4: 加载预处理数据...")
    try:
        from src.preprocess import main as preprocess_main
        result = preprocess_main()
        if result is None:
            log_message("数据预处理失败", "ERROR")
            return
        X_train, X_test, y_train, y_test = result
        log_message(f"✓ 数据加载成功")
        log_message(f"  测试集大小: {X_test.shape}")
    except Exception as e:
        log_message(f"数据加载失败: {e}", "ERROR")
        log_message("请先运行数据预处理: python src/preprocess.py", "ERROR")
        return
    
    # 加载XGBoost模型
    log_message("\n步骤 2/4: 加载XGBoost模型...")
    xgboost_model = load_model("xgboost")
    
    if xgboost_model is None:
        log_message("XGBoost模型加载失败", "ERROR")
        log_message("请先训练模型: python src/train_models.py", "ERROR")
        return
    log_message("✓ XGBoost模型加载成功")
    
    # 预测
    log_message("\n步骤 3/4: 在测试集上预测...")
    y_pred = xgboost_model.predict(X_test)
    log_message(f"✓ 预测完成，共 {len(y_pred)} 个样本")
    
    # 详细评估
    log_message("\n步骤 4/4: 生成评估报告...")
    
    # 1. 分类报告
    log_message("\n" + "="*60)
    log_message("XGBoost模型详细评估报告")
    log_message("="*60)
    
    print("\n📊 分类报告:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    # 保存分类报告
    report = classification_report(y_test, y_pred, 
                                   target_names=['Normal', 'Attack'],
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = REPORTS_DIR / "xgboost_classification_report.csv"
    report_df.to_csv(report_path)
    log_message(f"✓ 分类报告已保存: {report_path}")
    
    # 2. 混淆矩阵分析
    cm_metrics = calculate_confusion_matrix_metrics(y_test, y_pred)
    
    # 3. ROC曲线和AUC
    roc_auc = calculate_roc_auc(xgboost_model, X_test, y_test)
    if roc_auc:
        log_message(f"\n📈 ROC-AUC Score: {roc_auc:.4f}")
        # 保存AUC值
        with open(REPORTS_DIR / "xgboost_auc.txt", 'w') as f:
            f.write(f"AUC Score: {roc_auc:.4f}\n")
    
    # 4. 特征重要性
    feature_names = get_feature_names()
    importances, indices = print_feature_importance(xgboost_model, feature_names, top_n=20)
    
    # 保存特征重要性（修复长度不匹配问题）
    if importances is not None:
        # 确保特征名称数量与重要性数量匹配
        if len(feature_names) != len(importances):
            log_message(f"调整特征名称数量以匹配重要性数量: {len(importances)}")
            feature_names_adjusted = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names_adjusted = feature_names
        
        importance_df = pd.DataFrame({
            'feature': feature_names_adjusted[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv(REPORTS_DIR / "xgboost_feature_importance.csv", index=False)
        log_message(f"✓ 特征重要性已保存: {REPORTS_DIR / 'xgboost_feature_importance.csv'}")
    
    # 5. 误分类详细分析
    misclassified = np.where(y_pred != y_test)[0]
    log_message(f"\n🔍 误分类分析:")
    log_message(f"  误分类样本数: {len(misclassified)}/{len(y_test)} ({len(misclassified)/len(y_test)*100:.2f}%)")
    
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))
    
    log_message(f"  假阳性 (正常被误判为攻击): {false_positives}")
    log_message(f"  假阴性 (攻击被误判为正常): {false_negatives}")
    
    normal_count = np.sum(y_test == 0)
    attack_count = np.sum(y_test == 1)
    
    fp_rate = false_positives / normal_count if normal_count > 0 else 0
    fn_rate = false_negatives / attack_count if attack_count > 0 else 0
    
    log_message(f"  假阳性率: {fp_rate:.4f} ({fp_rate*100:.2f}%)")
    log_message(f"  假阴性率: {fn_rate:.4f} ({fn_rate*100:.2f}%)")
    
    # 6. 综合性能指标
    log_message(f"\n📊 综合性能指标:")
    log_message(f"  准确率 (Accuracy):  {cm_metrics['accuracy']:.4f}")
    log_message(f"  精确率 (Precision): {cm_metrics['precision']:.4f}")
    log_message(f"  召回率 (Recall):    {cm_metrics['recall']:.4f}")
    log_message(f"  F1分数 (F1-Score):  {cm_metrics['f1']:.4f}")
    
    # 7. 保存评估摘要
    summary = {
        'Model': 'XGBoost',
        'Accuracy': round(cm_metrics['accuracy'], 4),
        'Precision': round(cm_metrics['precision'], 4),
        'Recall': round(cm_metrics['recall'], 4),
        'F1-Score': round(cm_metrics['f1'], 4),
        'AUC': round(roc_auc, 4) if roc_auc else None,
        'False_Positive_Rate': round(fp_rate, 4),
        'False_Negative_Rate': round(fn_rate, 4),
        'Test_Samples': len(y_test)
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(REPORTS_DIR / "xgboost_evaluation_summary.csv", index=False)
    log_message(f"✓ 评估摘要已保存: {REPORTS_DIR / 'xgboost_evaluation_summary.csv'}")
    
    log_message("\n" + "="*60)
    log_message("✅ XGBoost模型评估完成！")
    log_message("="*60)
    
    return summary

def main():
    """主函数"""
    try:
        summary = evaluate_xgboost_model()
        if summary:
            print("\n🎯 评估结果摘要:")
            print(f"  模型: {summary['Model']}")
            print(f"  F1分数: {summary['F1-Score']:.4f}")
            print(f"  准确率: {summary['Accuracy']:.4f}")
            print(f"  召回率: {summary['Recall']:.4f}")
            if summary.get('AUC'):
                print(f"  AUC: {summary['AUC']:.4f}")
    except Exception as e:
        log_message(f"评估过程出错: {e}", "ERROR")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()