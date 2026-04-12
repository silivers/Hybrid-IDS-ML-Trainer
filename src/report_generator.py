"""
模型报告生成器 - 精简版XGBoost入侵检测系统报告
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report
from src.config import REPORTS_DIR
from src.utils import log_message


class XGBoostReportGenerator:
    """XGBoost模型报告生成器 - 精简版"""
    
    def __init__(self):
        self.report = {}
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def generate_complete_report(self, evaluation_results):
        """生成完整报告"""
        if evaluation_results is None:
            log_message("评估结果为空，无法生成报告", "ERROR")
            return None
        
        # 提取数据
        model = evaluation_results.get('model')
        X_test = evaluation_results.get('X_test')
        y_test = evaluation_results.get('y_test')
        y_pred = evaluation_results.get('y_pred')
        feature_importance = evaluation_results.get('feature_importance', {})
        metrics = evaluation_results.get('metrics', {})
        cm = evaluation_results.get('confusion_matrix', {})
        
        # 生成各模块
        self.add_model_info(model)
        self.add_data_info(X_test, y_test)
        self.add_performance_metrics(metrics)
        self.add_confusion_matrix(cm)
        self.add_classification_report(y_test, y_pred)
        self.add_roc_analysis(metrics)
        self.add_feature_importance(feature_importance)
        
        return self.report
    
    def add_model_info(self, model):
        """模型基本信息"""
        self.report['model_info'] = {
            'model_name': 'XGBoost',
            'tree_count': model.get_params().get('n_estimators', 'N/A') if model else 'N/A',
            'max_depth': model.get_params().get('max_depth', 'N/A') if model else 'N/A',
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def add_data_info(self, X_test, y_test):
        """数据信息"""
        if y_test is not None:
            normal_count = int(np.sum(y_test == 0))
            attack_count = int(np.sum(y_test == 1))
            total = len(y_test)
            
            self.report['data_info'] = {
                'total_samples': total,
                'normal_count': normal_count,
                'attack_count': attack_count,
                'attack_ratio': f"{attack_count / total * 100:.1f}%"
            }
    
    def add_performance_metrics(self, metrics):
        """性能指标"""
        self.report['performance_metrics'] = {
            'accuracy': f"{metrics.get('accuracy', 0):.2%}",
            'precision': f"{metrics.get('precision', 0):.2%}",
            'recall': f"{metrics.get('recall', 0):.2%}",
            'f1_score': f"{metrics.get('f1_score', 0):.3f}"
        }
    
    def add_confusion_matrix(self, cm):
        """混淆矩阵"""
        tn = cm.get('tn', 0)
        fp = cm.get('fp', 0)
        fn = cm.get('fn', 0)
        tp = cm.get('tp', 0)
        
        fp_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
        fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        self.report['confusion_matrix'] = {
            'true_negative': tn,  # 正确识别正常
            'false_positive': fp,  # 误报
            'false_negative': fn,  # 漏报
            'true_positive': tp,   # 正确识别攻击
            'false_positive_rate': f"{fp_rate:.2%}",
            'false_negative_rate': f"{fn_rate:.2%}"
        }
    
    def add_classification_report(self, y_test, y_pred):
        """分类报告"""
        if y_test is not None and y_pred is not None:
            report = classification_report(y_test, y_pred, 
                                          target_names=['正常', '攻击'],
                                          output_dict=True)
            
            self.report['classification_report'] = {
                'normal': {
                    'precision': f"{report['正常']['precision']:.2%}",
                    'recall': f"{report['正常']['recall']:.2%}",
                    'f1': f"{report['正常']['f1-score']:.3f}"
                },
                'attack': {
                    'precision': f"{report['攻击']['precision']:.2%}",
                    'recall': f"{report['攻击']['recall']:.2%}",
                    'f1': f"{report['攻击']['f1-score']:.3f}"
                },
                'accuracy': f"{report['accuracy']:.2%}"
            }
    
    def add_roc_analysis(self, metrics):
        """ROC分析"""
        roc_auc = metrics.get('auc')
        if roc_auc:
            self.report['roc_auc'] = f"{roc_auc:.3f}"
    
    def add_feature_importance(self, feature_importance_dict):
        """特征重要性 - 只保留前5个"""
        importances = feature_importance_dict.get('importances')
        indices = feature_importance_dict.get('indices')
        feature_names = feature_importance_dict.get('feature_names')
        
        if importances is not None and indices is not None:
            top_features = []
            for i in range(min(5, len(indices))):
                feat_name = feature_names[indices[i]] if feature_names and indices[i] < len(feature_names) else f"特征_{indices[i]}"
                top_features.append({
                    'rank': i + 1,
                    'name': feat_name,
                    'importance': f"{importances[indices[i]]:.2%}"
                })
            
            self.report['top_features'] = top_features
    
    def save_report(self):
        """保存Markdown报告"""
        report_path = REPORTS_DIR / f"xgboost_report_{self.timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self._format_markdown())
        
        log_message(f"报告已保存: {report_path}")
        return report_path
    
    def _format_markdown(self):
        """格式化为Markdown"""
        lines = []
        
        # 标题
        lines.append("# XGBoost入侵检测系统评估报告")
        lines.append(f"**生成时间：** {self.report['model_info']['generation_time']}")
        lines.append("")
        
        # 1. 模型信息
        lines.append("## 模型信息")
        lines.append(f"- **模型：** {self.report['model_info']['model_name']}")
        lines.append(f"- **决策树数量：** {self.report['model_info']['tree_count']}")
        lines.append(f"- **树的最大深度：** {self.report['model_info']['max_depth']}")
        lines.append("")
        
        # 2. 测试数据
        if 'data_info' in self.report:
            data = self.report['data_info']
            lines.append("## 测试数据")
            lines.append(f"- **总样本数：** {data['total_samples']}")
            lines.append(f"- **正常连接：** {data['normal_count']}")
            lines.append(f"- **攻击连接：** {data['attack_count']} ({data['attack_ratio']})")
            lines.append("")
        
        # 3. 性能指标
        lines.append("## 性能指标")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        for metric, value in self.report['performance_metrics'].items():
            metric_names = {
                'accuracy': '准确率',
                'precision': '精确率',
                'recall': '召回率',
                'f1_score': 'F1分数'
            }
            lines.append(f"| {metric_names.get(metric, metric)} | {value} |")
        lines.append("")
        
        # 4. 混淆矩阵
        cm = self.report['confusion_matrix']
        lines.append("## 混淆矩阵")
        lines.append("| | 预测正常 | 预测攻击 |")
        lines.append("|---|----------|----------|")
        lines.append(f"| **实际正常** | {cm['true_negative']} | {cm['false_positive']} |")
        lines.append(f"| **实际攻击** | {cm['false_negative']} | {cm['true_positive']} |")
        lines.append("")
        lines.append(f"- **误报率：** {cm['false_positive_rate']}")
        lines.append(f"- **漏报率：** {cm['false_negative_rate']}")
        lines.append("")
        
        # 5. 分类报告
        if 'classification_report' in self.report:
            cr = self.report['classification_report']
            lines.append("## 分类详情")
            lines.append("| 类型 | 精确率 | 召回率 | F1分数 |")
            lines.append("|------|--------|--------|--------|")
            lines.append(f"| 正常流量 | {cr['normal']['precision']} | {cr['normal']['recall']} | {cr['normal']['f1']} |")
            lines.append(f"| 攻击流量 | {cr['attack']['precision']} | {cr['attack']['recall']} | {cr['attack']['f1']} |")
            lines.append("")
            lines.append(f"**总体准确率：** {cr['accuracy']}")
            lines.append("")
        
        # 6. ROC-AUC
        if 'roc_auc' in self.report:
            lines.append("## ROC-AUC")
            lines.append(f"AUC分数：**{self.report['roc_auc']}**")
            lines.append("")
        
        # 7. 特征重要性
        if 'top_features' in self.report and self.report['top_features']:
            lines.append("## 重要特征 (Top 5)")
            lines.append("| 排名 | 特征 | 重要性 |")
            lines.append("|------|------|--------|")
            for feat in self.report['top_features']:
                lines.append(f"| {feat['rank']} | `{feat['name']}` | {feat['importance']} |")
            lines.append("")
        
        # 8. 简要总结
        lines.append("## 总结")
        acc = float(self.report['performance_metrics']['accuracy'].rstrip('%')) / 100
        if acc >= 0.95:
            lines.append("✅ 模型表现优秀，可以部署使用")
        elif acc >= 0.9:
            lines.append("👍 模型表现良好，基本满足需求")
        elif acc >= 0.8:
            lines.append("⚠️ 模型表现及格，建议配合人工审核")
        else:
            lines.append("🔴 模型表现需改进，暂不建议部署")
        
        lines.append("")
        lines.append("---")
        lines.append("*报告由 XGBoost入侵检测系统自动生成*")
        
        return "\n".join(lines)


def generate_model_report():
    """生成模型报告"""
    log_message("="*60)
    log_message("开始生成模型评估报告")
    log_message("="*60)
    
    try:
        from src.evaluate import evaluate_xgboost_model
        
        evaluation_results = evaluate_xgboost_model()
        
        if evaluation_results is None:
            log_message("评估失败，无法生成报告", "ERROR")
            return None
        
        generator = XGBoostReportGenerator()
        generator.generate_complete_report(evaluation_results)
        report_path = generator.save_report()
        
        log_message(f"报告生成完成: {report_path}")
        return generator.report
        
    except Exception as e:
        log_message(f"报告生成失败: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return None
    
def main():
    """主入口函数"""
    generate_model_report()

if __name__ == "__main__":
    generate_model_report()