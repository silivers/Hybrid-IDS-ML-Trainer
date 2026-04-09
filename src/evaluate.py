"""
模型评估脚本 - 详细评估最佳模型
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from src.config import FIGURES_DIR, REPORTS_DIR
from src.utils import (load_model, plot_confusion_matrix, plot_feature_importance, 
                       plot_roc_curve, log_message)

def get_feature_names():
    """获取特征名称"""
    # UNSW-NB15的49个特征（根据官方文档）
    feature_names = [
        'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 
        'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 
        'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 
        'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 
        'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 
        'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 
        'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
        'ct_dst_src_ltm', 'attack_cat', 'label'
    ]
    return feature_names

def evaluate_best_model():
    log_message("="*60)
    log_message("开始模型评估")
    log_message("="*60)
    
    # 获取预处理数据
    log_message("加载预处理数据...")
    try:
        from src.preprocess import main as preprocess_main
        X_train, X_test, y_train, y_test = preprocess_main()
    except:
        log_message("请先运行数据预处理", "ERROR")
        return
    
    # 加载最佳模型（XGBoost通常表现最好）
    log_message("加载最佳模型...")
    best_model = load_model("xgboost")
    
    if best_model is None:
        log_message("模型加载失败，请先训练模型", "ERROR")
        return
    
    # 预测
    y_pred = best_model.predict(X_test)
    
    # 详细分类报告
    log_message("\n" + "="*60)
    log_message("XGBoost 模型详细评估")
    log_message("="*60)
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    # 保存分类报告
    report = classification_report(y_test, y_pred, 
                                   target_names=['Normal', 'Attack'],
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORTS_DIR / "classification_report.csv")
    log_message(f"分类报告已保存: {REPORTS_DIR / 'classification_report.csv'}")
    
    # 混淆矩阵
    plot_confusion_matrix(y_test, y_pred, "XGBoost", 
                         FIGURES_DIR / "confusion_matrix.png")
    
    # ROC曲线
    roc_auc = plot_roc_curve(best_model, X_test, y_test, "XGBoost",
                            FIGURES_DIR / "roc_curve.png")
    
    if roc_auc:
        log_message(f"AUC Score: {roc_auc:.4f}")
    
    # 特征重要性（需要特征名称）
    feature_names = get_feature_names()
    # 移除标签列（因为X中不包含label和attack_cat）
    feature_names = [f for f in feature_names if f not in ['attack_cat', 'label']]
    
    plot_feature_importance(best_model, feature_names, top_n=20, 
                               save_path=FIGURES_DIR / "feature_importance.png")    
    # 分析误分类样本
    misclassified = np.where(y_pred != y_test)[0]
    log_message(f"\n误分类样本数: {len(misclassified)}/{len(y_test)} ({len(misclassified)/len(y_test)*100:.2f}%)")
    
    # 统计误分类类型
    false_positives = np.sum((y_pred == 1) & (y_test == 0))  # 假阳性
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))  # 假阴性
    
    log_message(f"假阳性 (正常被误判为攻击): {false_positives}")
    log_message(f"假阴性 (攻击被误判为正常): {false_negatives}")
    
    # 计算误分类率
    normal_count = np.sum(y_test == 0)
    attack_count = np.sum(y_test == 1)
    
    fp_rate = false_positives / normal_count if normal_count > 0 else 0
    fn_rate = false_negatives / attack_count if attack_count > 0 else 0
    
    log_message(f"假阳性率: {fp_rate:.4f} ({fp_rate*100:.2f}%)")
    log_message(f"假阴性率: {fn_rate:.4f} ({fn_rate*100:.2f}%)")
    
    log_message("\n模型评估完成！")

if __name__ == "__main__":
    evaluate_best_model()