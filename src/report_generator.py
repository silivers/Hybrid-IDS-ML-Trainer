"""
模型报告生成器 - 生成完整的XGBoost入侵检测系统报告 (Markdown格式)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from src.config import REPORTS_DIR, XGB_PARAMS
from src.utils import log_message


class XGBoostReportGenerator:
    """XGBoost模型报告生成器 - Markdown格式"""
    
    def __init__(self):
        self.report = {}
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def generate_complete_report(self, evaluation_results):
        """
        基于评估结果生成完整报告
        
        Args:
            evaluation_results: 从 evaluate_xgboost_model() 返回的结果字典
        """
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
        roc_curve = evaluation_results.get('roc_curve', {})
        
        # 1. 模型信息
        self.add_model_info(model)
        
        # 2. 参数配置
        self.add_parameter_configuration()
        
        # 3. 数据信息
        self.add_data_info(X_test, y_test)
        
        # 4. 性能指标
        self.add_performance_metrics(metrics)
        
        # 5. 混淆矩阵分析
        self.add_confusion_matrix_analysis(cm, y_test, y_pred)
        
        # 6. 分类报告
        self.add_classification_report(y_test, y_pred)
        
        # 7. ROC和AUC
        self.add_roc_analysis(roc_curve, metrics)
        
        # 8. 特征重要性
        self.add_feature_importance_analysis(feature_importance)
        
        # 9. 错误分析
        self.add_error_analysis(y_test, y_pred)
        
        # 10. 部署建议
        self.add_deployment_recommendations()
        
        return self.report
    
    def add_model_info(self, model):
        """添加模型基本信息"""
        self.report['model_info'] = {
            'model_type': 'XGBoost (eXtreme Gradient Boosting)',
            'model_version': 'xgboost.pkl',
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_architecture': 'Gradient Boosting Decision Trees (GBDT)',
            'tree_count': model.get_params().get('n_estimators', 'N/A') if model else 'N/A',
            'max_tree_depth': model.get_params().get('max_depth', 'N/A') if model else 'N/A'
        }
    
    def add_parameter_configuration(self):
        """添加参数配置和注解"""
        param_annotations = {
            'n_estimators': {
                'value': XGB_PARAMS.get('n_estimators', 200),
                'description': '决策树的数量',
                'impact': '更多的树可以提高精度，但增加训练时间和模型大小',
                'recommended_range': '100-500',
                'current_setting': '200 - 平衡精度和性能'
            },
            'max_depth': {
                'value': XGB_PARAMS.get('max_depth', 10),
                'description': '每棵决策树的最大深度',
                'impact': '越深的树可以学习更复杂的模式，但容易过拟合',
                'recommended_range': '3-15',
                'current_setting': '10 - 适中的复杂度'
            },
            'learning_rate': {
                'value': XGB_PARAMS.get('learning_rate', 0.05),
                'description': '学习率（步长）',
                'impact': '较小的学习率需要更多的树，但通常精度更高',
                'recommended_range': '0.01-0.3',
                'current_setting': '0.05 - 平衡收敛速度和精度'
            },
            'subsample': {
                'value': XGB_PARAMS.get('subsample', 0.8),
                'description': '每棵树的训练样本采样比例',
                'impact': '小于1.0可以防止过拟合，增加泛化能力',
                'recommended_range': '0.6-1.0',
                'current_setting': '0.8 - 80%的样本用于训练每棵树'
            },
            'colsample_bytree': {
                'value': XGB_PARAMS.get('colsample_bytree', 0.8),
                'description': '每棵树的特征采样比例',
                'impact': '特征采样可以增加多样性，防止过拟合',
                'recommended_range': '0.6-1.0',
                'current_setting': '0.8 - 80%的特征用于训练每棵树'
            },
            'random_state': {
                'value': XGB_PARAMS.get('random_state', 42),
                'description': '随机种子，确保结果可重复',
                'impact': '固定随机数保证实验结果一致',
                'recommended_range': '任意整数',
                'current_setting': '42 - 固定的随机种子'
            },
            'eval_metric': {
                'value': XGB_PARAMS.get('eval_metric', 'logloss'),
                'description': '模型评估指标',
                'impact': '影响模型优化的方向',
                'recommended_range': 'logloss, error, auc',
                'current_setting': 'logloss - 对数损失函数'
            }
        }
        
        self.report['parameter_configuration'] = param_annotations
    
    def add_data_info(self, X_test, y_test):
        """添加数据信息"""
        if X_test is not None and y_test is not None:
            self.report['data_info'] = {
                'test_samples': len(y_test),
                'features_count': X_test.shape[1] if hasattr(X_test, 'shape') else 'N/A',
                'normal_count': int(np.sum(y_test == 0)),
                'attack_count': int(np.sum(y_test == 1)),
                'attack_ratio': f"{np.sum(y_test == 1) / len(y_test) * 100:.2f}%",
                'feature_range': f"[{X_test.min():.4f}, {X_test.max():.4f}]" if hasattr(X_test, 'min') else 'N/A'
            }
    
    def add_performance_metrics(self, metrics):
        """添加性能指标"""
        self.report['performance_metrics'] = {
            'accuracy': {
                'value': f"{metrics.get('accuracy', 0):.4f}",
                'description': '准确率 - 正确预测的比例',
                'interpretation': f"模型正确预测了 {metrics.get('accuracy', 0)*100:.2f}% 的样本"
            },
            'precision': {
                'value': f"{metrics.get('precision', 0):.4f}",
                'description': '精确率 - 预测为攻击中真正的攻击比例',
                'interpretation': f"当模型预测为攻击时，有 {metrics.get('precision', 0)*100:.2f}% 的概率是正确的"
            },
            'recall': {
                'value': f"{metrics.get('recall', 0):.4f}",
                'description': '召回率 - 真正的攻击中被检测出的比例',
                'interpretation': f"模型能够检测出 {metrics.get('recall', 0)*100:.2f}% 的真实攻击"
            },
            'f1_score': {
                'value': f"{metrics.get('f1_score', 0):.4f}",
                'description': 'F1分数 - 精确率和召回率的调和平均',
                'interpretation': f"模型的综合性能评分为 {metrics.get('f1_score', 0):.4f}"
            }
        }
    
    def add_confusion_matrix_analysis(self, cm_dict, y_test, y_pred):
        """添加混淆矩阵分析"""
        tn = cm_dict.get('tn', 0)
        fp = cm_dict.get('fp', 0)
        fn = cm_dict.get('fn', 0)
        tp = cm_dict.get('tp', 0)
        
        fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        self.report['confusion_matrix'] = {
            'matrix_values': {
                'True_Negatives': {'value': int(tn), 'meaning': '正常流量被正确识别为正常'},
                'False_Positives': {'value': int(fp), 'meaning': '正常流量被误判为攻击（误报）'},
                'False_Negatives': {'value': int(fn), 'meaning': '攻击流量被漏判为正常（漏报）'},
                'True_Positives': {'value': int(tp), 'meaning': '攻击流量被正确识别为攻击'}
            },
            'derived_metrics': {
                'false_positive_rate': f"{fp / (tn + fp) if (tn + fp) > 0 else 0:.4f}",
                'false_negative_rate': f"{fn / (tp + fn) if (tp + fn) > 0 else 0:.4f}",
                'true_positive_rate': f"{tp / (tp + fn) if (tp + fn) > 0 else 0:.4f}",
                'true_negative_rate': f"{tn / (tn + fp) if (tn + fp) > 0 else 0:.4f}"
            },
            'security_impact': {
                'security_level': '优秀' if fn_rate < 0.03 else '良好' if fn_rate < 0.05 else '一般' if fn_rate < 0.1 else '需改进'
            }
        }
    
    def add_classification_report(self, y_test, y_pred):
        """添加分类报告"""
        if y_test is not None and y_pred is not None:
            report = classification_report(y_test, y_pred, 
                                          target_names=['Normal', 'Attack'],
                                          output_dict=True)
            
            self.report['classification_report'] = {
                'normal_class': {
                    'precision': f"{report['Normal']['precision']:.4f}",
                    'recall': f"{report['Normal']['recall']:.4f}",
                    'f1_score': f"{report['Normal']['f1-score']:.4f}",
                    'support': int(report['Normal']['support'])
                },
                'attack_class': {
                    'precision': f"{report['Attack']['precision']:.4f}",
                    'recall': f"{report['Attack']['recall']:.4f}",
                    'f1_score': f"{report['Attack']['f1-score']:.4f}",
                    'support': int(report['Attack']['support'])
                },
                'overall_accuracy': f"{report['accuracy']:.4f}"
            }
    
    def add_roc_analysis(self, roc_curve, metrics):
        """添加ROC曲线分析"""
        roc_auc = metrics.get('auc')
        
        if roc_auc:
            self.report['roc_analysis'] = {
                'auc_score': {
                    'value': f"{roc_auc:.4f}",
                    'interpretation': self._interpret_auc(roc_auc)
                },
                'model_performance_level': self._get_auc_level(roc_auc)
            }
        else:
            self.report['roc_analysis'] = {
                'auc_score': {'value': 'N/A', 'interpretation': '无法计算AUC'},
                'model_performance_level': 'N/A'
            }
    
    def add_feature_importance_analysis(self, feature_importance_dict):
        """添加特征重要性分析"""
        importances = feature_importance_dict.get('importances')
        indices = feature_importance_dict.get('indices')
        feature_names = feature_importance_dict.get('feature_names')
        
        if importances is not None and indices is not None:
            top_features = []
            cumulative_importance = 0
            
            for i in range(min(10, len(indices))):
                cumulative_importance += importances[indices[i]]
                top_features.append({
                    'rank': i + 1,
                    'feature_name': feature_names[indices[i]] if feature_names and indices[i] < len(feature_names) else f"feature_{indices[i]}",
                    'importance': f"{importances[indices[i]]:.4f}",
                    'cumulative_importance': f"{cumulative_importance:.4f}"
                })
            
            # 计算累计重要性
            cumulative = np.cumsum(importances[indices])
            features_to_80 = np.searchsorted(cumulative, 0.8) + 1
            
            self.report['feature_importance'] = {
                'top_10_features': top_features,
                'total_features': len(importances),
                'analysis': {
                    'most_important_feature': f"{top_features[0]['feature_name']} (重要性: {top_features[0]['importance']})" if top_features else "N/A",
                    'features_to_reach_80_percent': features_to_80
                }
            }
        else:
            self.report['feature_importance'] = {
                'top_10_features': [],
                'total_features': 0,
                'analysis': {'most_important_feature': '无法计算特征重要性'}
            }
    
    def add_error_analysis(self, y_test, y_pred):
        """添加错误分析"""
        if y_test is not None and y_pred is not None:
            misclassified = np.where(y_pred != y_test)[0]
            false_positives = np.where((y_pred == 1) & (y_test == 0))[0]
            false_negatives = np.where((y_pred == 0) & (y_test == 1))[0]
            
            self.report['error_analysis'] = {
                'total_errors': len(misclassified),
                'error_rate': f"{len(misclassified) / len(y_test) * 100:.2f}%",
                'false_positives_count': len(false_positives),
                'false_negatives_count': len(false_negatives),
                'improvement_suggestions': self._get_improvement_suggestions(len(false_positives), len(false_negatives))
            }
    
    def add_deployment_recommendations(self):
        """添加部署建议"""
        self.report['deployment_recommendations'] = {
            'model_files_required': [
                'models/xgboost.pkl - 核心模型文件',
                'models/xgboost_label_encoders.pkl - 类别编码器',
                'models/xgboost_scaler.pkl - 特征标准化器'
            ],
            'performance_expectations': {
                'inference_time': '约 0.05秒/样本',
                'memory_usage': '约 80-100 MB',
                'cpu_requirement': '建议 4核心以上',
                'throughput': '约 20,000 样本/秒（批量处理）'
            },
            'alert_thresholds': {
                'high_confidence': '置信度 > 0.9 → 立即告警',
                'medium_confidence': '0.7 < 置信度 ≤ 0.9 → 需要人工确认',
                'low_confidence': '置信度 ≤ 0.7 → 仅记录日志'
            }
        }
    
    def save_report(self, format='md'):
        """保存报告到文件 (Markdown格式)"""
        report_path = REPORTS_DIR / f"xgboost_complete_report_{self.timestamp}"
        
        # 保存为Markdown格式
        if format in ['md', 'both']:
            md_path = report_path.with_suffix('.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(self._format_report_as_markdown())
            log_message(f"📄 Markdown报告已保存: {md_path}")
        
        # 可选：同时保存JSON格式
        if format == 'both':
            import json
            json_path = report_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.report, f, indent=2, ensure_ascii=False)
            log_message(f"📄 JSON报告已保存: {json_path}")
        
        return report_path
    
    def _interpret_auc(self, auc_score):
        """解释AUC分数"""
        if auc_score >= 0.9:
            return "优秀 - 模型具有极佳的区分能力"
        elif auc_score >= 0.8:
            return "良好 - 模型具有较好的区分能力"
        elif auc_score >= 0.7:
            return "可接受 - 模型具有一定的区分能力"
        else:
            return "需改进 - 模型区分能力较弱"
    
    def _get_auc_level(self, auc_score):
        """获取AUC等级"""
        if auc_score >= 0.9:
            return "优秀 (Excellent)"
        elif auc_score >= 0.8:
            return "良好 (Good)"
        elif auc_score >= 0.7:
            return "可接受 (Acceptable)"
        else:
            return "需改进 (Need Improvement)"
    
    def _get_improvement_suggestions(self, fp_count, fn_count):
        """获取改进建议"""
        suggestions = []
        
        if fp_count > fn_count:
            suggestions.append("假阳性较多 → 建议提高分类阈值或增加正常样本训练")
        elif fn_count > fp_count:
            suggestions.append("假阴性较多 → 建议降低分类阈值或增加攻击样本训练")
        else:
            suggestions.append("误报和漏报平衡 → 当前配置合理")
        
        if fp_count + fn_count > 1000:
            suggestions.append("总体错误较多 → 建议收集更多训练数据或调整模型参数")
        
        suggestions.append("定期重新训练模型以保持检测能力")
        
        return suggestions
    
    def _format_report_as_markdown(self):
        """将报告格式化为Markdown"""
        lines = []
        
        # 标题
        lines.append("# XGBoost入侵检测系统 - 完整模型评估报告")
        lines.append("")
        lines.append(f"**生成时间:** {self.report['model_info']['generation_time']}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # 1. 模型信息
        lines.append("## 1. 模型信息")
        lines.append("")
        lines.append("| 属性 | 值 |")
        lines.append("|------|-----|")
        for key, value in self.report['model_info'].items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
        
        # 2. 参数配置
        lines.append("## 2. 模型参数配置")
        lines.append("")
        lines.append("| 参数 | 当前值 | 说明 | 影响 | 建议范围 |")
        lines.append("|------|--------|------|------|----------|")
        for param, info in self.report['parameter_configuration'].items():
            lines.append(f"| {param} | {info['value']} | {info['description']} | {info['impact']} | {info['recommended_range']} |")
        lines.append("")
        
        # 3. 数据信息
        if 'data_info' in self.report:
            lines.append("## 3. 测试数据信息")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            for key, value in self.report['data_info'].items():
                lines.append(f"| {key} | {value} |")
            lines.append("")
        
        # 4. 性能指标
        lines.append("## 4. 性能指标")
        lines.append("")
        lines.append("| 指标 | 值 | 说明 | 解读 |")
        lines.append("|------|-----|------|------|")
        for metric, info in self.report['performance_metrics'].items():
            lines.append(f"| **{metric.upper()}** | {info['value']} | {info['description']} | {info['interpretation']} |")
        lines.append("")
        
        # 5. 混淆矩阵
        lines.append("## 5. 混淆矩阵分析")
        lines.append("")
        lines.append("### 5.1 混淆矩阵数值")
        lines.append("")
        lines.append("| 类别 | 数值 | 含义 |")
        lines.append("|------|------|------|")
        for key, value in self.report['confusion_matrix']['matrix_values'].items():
            lines.append(f"| {key} | {value['value']} | {value['meaning']} |")
        lines.append("")
        
        lines.append("### 5.2 衍生指标")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        for key, value in self.report['confusion_matrix']['derived_metrics'].items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
        
        lines.append("### 5.3 安全影响评估")
        lines.append("")
        lines.append(f"**安全等级:** {self.report['confusion_matrix']['security_impact']['security_level']}")
        lines.append("")
        
        # 6. 分类报告
        if 'classification_report' in self.report:
            lines.append("## 6. 分类报告")
            lines.append("")
            lines.append("### 6.1 各类别性能")
            lines.append("")
            lines.append("| 类别 | 精确率 | 召回率 | F1分数 | 样本数 |")
            lines.append("|------|--------|--------|--------|--------|")
            for class_name in ['normal_class', 'attack_class']:
                name = "正常 (Normal)" if class_name == 'normal_class' else "攻击 (Attack)"
                info = self.report['classification_report'][class_name]
                lines.append(f"| {name} | {info['precision']} | {info['recall']} | {info['f1_score']} | {info['support']} |")
            lines.append("")
            lines.append(f"**总体准确率:** {self.report['classification_report']['overall_accuracy']}")
            lines.append("")
        
        # 7. ROC和AUC
        lines.append("## 7. ROC曲线分析")
        lines.append("")
        lines.append(f"**AUC分数:** {self.report['roc_analysis']['auc_score']['value']}")
        lines.append("")
        lines.append(f"**评价:** {self.report['roc_analysis']['auc_score']['interpretation']}")
        lines.append("")
        lines.append(f"**模型等级:** {self.report['roc_analysis']['model_performance_level']}")
        lines.append("")
        
        # 8. 特征重要性
        lines.append("## 8. 特征重要性分析")
        lines.append("")
        lines.append(f"**总特征数:** {self.report['feature_importance']['total_features']}")
        lines.append("")
        lines.append("### 8.1 Top 10 重要特征")
        lines.append("")
        lines.append("| 排名 | 特征名称 | 重要性 | 累计重要性 |")
        lines.append("|------|----------|--------|------------|")
        for feat in self.report['feature_importance']['top_10_features']:
            lines.append(f"| {feat['rank']} | `{feat['feature_name']}` | {feat['importance']} | {feat['cumulative_importance']} |")
        lines.append("")
        
        analysis = self.report['feature_importance']['analysis']
        lines.append(f"**最重要特征:** {analysis['most_important_feature']}")
        lines.append("")
        lines.append(f"**达到80%重要性需要的特征数:** {analysis['features_to_reach_80_percent']}")
        lines.append("")
        
        # 9. 错误分析
        if 'error_analysis' in self.report:
            lines.append("## 9. 错误分析")
            lines.append("")
            lines.append(f"- **总错误数:** {self.report['error_analysis']['total_errors']}")
            lines.append(f"- **错误率:** {self.report['error_analysis']['error_rate']}")
            lines.append(f"- **假阳性 (误报):** {self.report['error_analysis']['false_positives_count']}")
            lines.append(f"- **假阴性 (漏报):** {self.report['error_analysis']['false_negatives_count']}")
            lines.append("")
            lines.append("### 改进建议")
            lines.append("")
            for suggestion in self.report['error_analysis']['improvement_suggestions']:
                lines.append(f"- {suggestion}")
            lines.append("")
        
        # 10. 部署建议
        lines.append("## 10. 部署建议")
        lines.append("")
        lines.append("### 10.1 必需文件")
        lines.append("")
        for file in self.report['deployment_recommendations']['model_files_required']:
            lines.append(f"- `{file}`")
        lines.append("")
        
        lines.append("### 10.2 性能预期")
        lines.append("")
        lines.append("| 指标 | 预期值 |")
        lines.append("|------|--------|")
        for key, value in self.report['deployment_recommendations']['performance_expectations'].items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
        
        lines.append("### 10.3 告警阈值建议")
        lines.append("")
        lines.append("| 置信度级别 | 处理方式 |")
        lines.append("|------------|----------|")
        for key, value in self.report['deployment_recommendations']['alert_thresholds'].items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
        
        # 页脚
        lines.append("---")
        lines.append("")
        lines.append("*报告由 XGBoost入侵检测系统自动生成*")
        
        return "\n".join(lines)


def generate_model_report():
    """生成完整的模型报告"""
    log_message("="*60)
    log_message("开始生成完整模型报告 (Markdown格式)")
    log_message("="*60)
    
    try:
        # 导入评估模块
        from src.evaluate import evaluate_xgboost_model
        
        # 获取评估结果
        evaluation_results = evaluate_xgboost_model()
        
        if evaluation_results is None:
            log_message("评估失败，无法生成报告", "ERROR")
            return None
        
        # 生成报告
        generator = XGBoostReportGenerator()
        report = generator.generate_complete_report(evaluation_results)
        
        # 保存报告 (Markdown格式)
        report_path = generator.save_report(format='md')
        
        log_message(f"\n✅ 完整报告已生成!")
        log_message(f"📄 报告位置: {report_path}")
        
        return report
        
    except Exception as e:
        log_message(f"报告生成失败: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    generate_model_report()