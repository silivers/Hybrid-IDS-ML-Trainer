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
    """XGBoost模型报告生成器 - Markdown格式（通俗易懂版）"""
    
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
        
        # 1. 模型信息
        self.add_model_info(model)
        
        # 2. 参数配置（简化版）
        self.add_parameter_configuration_simple()
        
        # 3. 数据信息
        self.add_data_info(X_test, y_test)
        
        # 4. 性能指标（通俗解释）
        self.add_performance_metrics_simple(metrics)
        
        # 5. 混淆矩阵分析（通俗解释）
        self.add_confusion_matrix_analysis_simple(cm)
        
        # 6. 分类报告
        self.add_classification_report_simple(y_test, y_pred)
        
        # 7. ROC和AUC
        self.add_roc_analysis_simple(metrics)
        
        # 8. 特征重要性（简化）
        self.add_feature_importance_simple(feature_importance)
        
        # 9. 错误分析
        self.add_error_analysis_simple(y_test, y_pred)
        
        # 10. 部署建议（通俗版）
        self.add_deployment_recommendations_simple()
        
        return self.report
    
    def add_model_info(self, model):
        """添加模型基本信息"""
        self.report['model_info'] = {
            'model_name': 'XGBoost（极致梯度提升）',
            'model_type': '决策树集成模型',
            'what_it_does': '通过组合多棵决策树来识别网络攻击',
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tree_count': model.get_params().get('n_estimators', 'N/A') if model else 'N/A',
            'tree_depth': model.get_params().get('max_depth', 'N/A') if model else 'N/A'
        }
    
    def add_parameter_configuration_simple(self):
        """添加参数配置（通俗版）"""
        param_annotations = {
            'n_estimators': {
                'value': XGB_PARAMS.get('n_estimators', 200),
                'simple_description': '决策树数量',
                'what_it_means': '模型使用了200棵决策树。更多的树能让模型学得更好，但训练时间也会更长。'
            },
            'max_depth': {
                'value': XGB_PARAMS.get('max_depth', 10),
                'simple_description': '树的最大深度',
                'what_it_means': '每棵决策树最多可以有10层。越深的树能学习更复杂的模式，但也可能记住太多细节（过拟合）。'
            },
            'learning_rate': {
                'value': XGB_PARAMS.get('learning_rate', 0.05),
                'simple_description': '学习率',
                'what_it_means': '0.05的学习率意味着每棵树对最终结果的贡献较小，需要更多树来达到好效果，但通常更稳定。'
            },
            'subsample': {
                'value': XGB_PARAMS.get('subsample', 0.8),
                'simple_description': '样本采样比例',
                'what_it_means': '每棵树只使用80%的训练数据，这有助于防止模型过拟合，提高泛化能力。'
            },
            'colsample_bytree': {
                'value': XGB_PARAMS.get('colsample_bytree', 0.8),
                'simple_description': '特征采样比例',
                'what_it_means': '每棵树只随机选择80%的特征来训练，这增加了树的多样性，让模型更健壮。'
            }
        }
        
        self.report['parameter_configuration'] = param_annotations
    
    def add_data_info(self, X_test, y_test):
        """添加数据信息"""
        if X_test is not None and y_test is not None:
            normal_count = int(np.sum(y_test == 0))
            attack_count = int(np.sum(y_test == 1))
            total = len(y_test)
            
            self.report['data_info'] = {
                'total_samples': total,
                'normal_count': normal_count,
                'attack_count': attack_count,
                'attack_percentage': f"{attack_count / total * 100:.1f}%",
                'normal_percentage': f"{normal_count / total * 100:.1f}%",
                'features_count': X_test.shape[1] if hasattr(X_test, 'shape') else 'N/A'
            }
    
    def add_performance_metrics_simple(self, metrics):
        """添加性能指标（通俗解释）"""
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1_score', 0)
        
        # 生成通俗易懂的评级
        def get_rating(score, thresholds=[0.9, 0.8, 0.7]):
            if score >= thresholds[0]:
                return "优秀 ⭐⭐⭐"
            elif score >= thresholds[1]:
                return "良好 ⭐⭐"
            elif score >= thresholds[2]:
                return "及格 ⭐"
            else:
                return "需改进 ⚠️"
        
        self.report['performance_metrics'] = {
            'accuracy': {
                'value': f"{accuracy:.2%}",
                'rating': get_rating(accuracy),
                'simple_explanation': f"模型判断了 {accuracy:.2%} 的流量，判断是正确的。"
            },
            'precision': {
                'value': f"{precision:.2%}",
                'rating': get_rating(precision),
                'simple_explanation': f"当模型说'这是攻击'时，有 {precision:.2%} 的概率真的是攻击。"
            },
            'recall': {
                'value': f"{recall:.2%}",
                'rating': get_rating(recall),
                'simple_explanation': f"在所有真实的攻击中，模型成功发现了 {recall:.2%}。"
            },
            'f1_score': {
                'value': f"{f1:.3f}",
                'rating': get_rating(f1, [0.9, 0.8, 0.7]),
                'simple_explanation': f"综合评分 {f1:.3f}（最高为1），平衡了精确率和召回率。"
            }
        }
    
    def add_confusion_matrix_analysis_simple(self, cm_dict):
        """添加混淆矩阵分析（通俗解释）"""
        tn = cm_dict.get('tn', 0)
        fp = cm_dict.get('fp', 0)
        fn = cm_dict.get('fn', 0)
        tp = cm_dict.get('tp', 0)
        
        total = tn + fp + fn + tp
        fp_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
        fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        # 安全等级评估
        if fn_rate < 0.03:
            security_level = "优秀 🟢"
            security_msg = "漏报率极低，几乎不会放过任何攻击"
        elif fn_rate < 0.05:
            security_level = "良好 🟡"
            security_msg = "漏报率较低，安全风险可控"
        elif fn_rate < 0.10:
            security_level = "一般 🟠"
            security_msg = "有一定漏报风险，建议关注"
        else:
            security_level = "需改进 🔴"
            security_msg = "漏报率较高，建议优化模型"
        
        self.report['confusion_matrix'] = {
            'what_is_it': '混淆矩阵告诉我们模型的判断有多准确',
            'results': {
                'correct_normal': {
                    'count': tn,
                    'meaning': f'正常流量被正确识别为正常（共{tn}个）'
                },
                'false_alarm': {
                    'count': fp,
                    'meaning': f'正常流量被误判为攻击（误报，共{fp}个，占正常流量的{fp_rate:.1%}）'
                },
                'missed_attack': {
                    'count': fn,
                    'meaning': f'攻击流量被漏掉（漏报，共{fn}个，占攻击流量的{fn_rate:.1%}）'
                },
                'caught_attack': {
                    'count': tp,
                    'meaning': f'攻击流量被正确识别（共{tp}个）'
                }
            },
            'security_assessment': {
                'level': security_level,
                'message': security_msg,
                'total_tested': total,
                'false_alarm_rate': f"{fp_rate:.2%}",
                'missed_rate': f"{fn_rate:.2%}"
            }
        }
    
    def add_classification_report_simple(self, y_test, y_pred):
        """添加分类报告（简化版）"""
        if y_test is not None and y_pred is not None:
            report = classification_report(y_test, y_pred, 
                                          target_names=['正常流量', '攻击流量'],
                                          output_dict=True)
            
            normal_info = report['正常流量']
            attack_info = report['攻击流量']
            
            self.report['classification_report'] = {
                'normal': {
                    'precision': f"{normal_info['precision']:.2%}",
                    'recall': f"{normal_info['recall']:.2%}",
                    'f1': f"{normal_info['f1-score']:.3f}",
                    'sample_count': int(normal_info['support'])
                },
                'attack': {
                    'precision': f"{attack_info['precision']:.2%}",
                    'recall': f"{attack_info['recall']:.2%}",
                    'f1': f"{attack_info['f1-score']:.3f}",
                    'sample_count': int(attack_info['support'])
                },
                'overall_accuracy': f"{report['accuracy']:.2%}"
            }
    
    def add_roc_analysis_simple(self, metrics):
        """添加ROC曲线分析（通俗版）"""
        roc_auc = metrics.get('auc')
        
        if roc_auc:
            if roc_auc >= 0.9:
                rating = "优秀 🌟🌟🌟"
                msg = "模型区分正常流量和攻击流量的能力非常出色"
            elif roc_auc >= 0.8:
                rating = "良好 🌟🌟"
                msg = "模型有较好的区分能力，可以信赖"
            elif roc_auc >= 0.7:
                rating = "及格 🌟"
                msg = "模型有一定区分能力，建议配合其他措施使用"
            else:
                rating = "需改进 ⚠️"
                msg = "模型区分能力较弱，建议优化"
            
            self.report['roc_analysis'] = {
                'what_is_auc': 'AUC分数衡量模型区分"正常"和"攻击"的能力，越接近1越好',
                'auc_score': f"{roc_auc:.3f}",
                'rating': rating,
                'interpretation': msg
            }
        else:
            self.report['roc_analysis'] = {
                'auc_score': '无法计算',
                'interpretation': '模型不支持概率预测'
            }
    
    def add_feature_importance_simple(self, feature_importance_dict):
        """添加特征重要性（简化版）"""
        importances = feature_importance_dict.get('importances')
        indices = feature_importance_dict.get('indices')
        feature_names = feature_importance_dict.get('feature_names')
        
        if importances is not None and indices is not None:
            top_features = []
            feature_meanings = {
                'dur': '连接持续时间',
                'sbytes': '源到目的地的字节数',
                'dbytes': '目的地到源的字节数',
                'proto': '协议类型',
                'service': '服务类型',
                'state': '连接状态',
                'sload': '源端负载',
                'dload': '目的端负载',
                'spkts': '源端数据包数',
                'dpkts': '目的端数据包数',
                'tcprtt': 'TCP往返时间',
                'synack': 'SYN-ACK时间',
                'ackdat': 'ACK数据时间'
            }
            
            for i in range(min(10, len(indices))):
                feat_name = feature_names[indices[i]] if feature_names and indices[i] < len(feature_names) else f"特征_{indices[i]}"
                meaning = feature_meanings.get(feat_name, '网络流量特征')
                top_features.append({
                    'rank': i + 1,
                    'feature_name': feat_name,
                    'meaning': meaning,
                    'importance': f"{importances[indices[i]]:.2%}"
                })
            
            self.report['feature_importance'] = {
                'explanation': '特征重要性告诉我们哪些网络特征对判断攻击最有帮助',
                'top_features': top_features,
                'note': '重要性越高，说明这个特征对判断攻击越关键'
            }
        else:
            self.report['feature_importance'] = {
                'top_features': [],
                'note': '无法计算特征重要性'
            }
    
    def add_error_analysis_simple(self, y_test, y_pred):
        """添加错误分析（通俗版）"""
        if y_test is not None and y_pred is not None:
            misclassified = np.where(y_pred != y_test)[0]
            false_positives = np.where((y_pred == 1) & (y_test == 0))[0]
            false_negatives = np.where((y_pred == 0) & (y_test == 1))[0]
            
            error_rate = len(misclassified) / len(y_test)
            
            # 生成改进建议
            suggestions = []
            if len(false_positives) > len(false_negatives):
                suggestions.append("🔧 误报较多 → 建议适当提高判断阈值，减少正常流量被误判")
            elif len(false_negatives) > len(false_positives):
                suggestions.append("⚠️ 漏报较多 → 建议适当降低判断阈值，捕获更多攻击")
            else:
                suggestions.append("✓ 误报和漏报相对平衡")
            
            if error_rate > 0.1:
                suggestions.append("📊 总体错误率偏高 → 建议收集更多训练数据或调整模型参数")
            
            suggestions.append("🔄 建议定期（如每周）用新数据重新训练模型，保持检测能力")
            
            self.report['error_analysis'] = {
                'summary': {
                    'error_rate': f"{error_rate:.2%}",
                    'error_count': len(misclassified),
                    'total_tested': len(y_test)
                },
                'error_types': {
                    'false_positives': {
                        'count': len(false_positives),
                        'percentage': f"{len(false_positives)/len(y_test)*100:.2f}%",
                        'explanation': '正常流量被误判为攻击（误报）'
                    },
                    'false_negatives': {
                        'count': len(false_negatives),
                        'percentage': f"{len(false_negatives)/len(y_test)*100:.2f}%",
                        'explanation': '攻击流量被漏掉（漏报）- 这是最需要关注的问题'
                    }
                },
                'improvement_suggestions': suggestions
            }
    
    def add_deployment_recommendations_simple(self):
        """添加部署建议（通俗版）"""
        self.report['deployment_recommendations'] = {
            'files_needed': [
                '📁 models/xgboost.pkl - 训练好的模型文件',
                '📁 models/xgboost_label_encoders.pkl - 数据编码器',
                '📁 models/xgboost_scaler.pkl - 数据标准化器'
            ],
            'performance': {
                'speed': '约 0.05秒 判断一个网络连接',
                'batch_speed': '每秒可处理约 20,000 个连接（批量处理）',
                'memory': '约 80-100 MB 内存',
                'cpu': '建议使用4核心以上的CPU'
            },
            'alert_suggestions': {
                'high_confidence': '置信度 > 90% → 立即告警，人工确认',
                'medium_confidence': '70% - 90% → 记录日志，定期复查',
                'low_confidence': '置信度 < 70% → 仅记录，不告警'
            },
            'maintenance': {
                'retrain_frequency': '建议每周或每月用新数据重新训练',
                'monitoring': '定期监控误报率和漏报率的变化趋势'
            }
        }
    
    def save_report(self, format='md'):
        """保存报告到文件 (Markdown格式)"""
        report_path = REPORTS_DIR / f"xgboost_report_{self.timestamp}"
        
        if format in ['md', 'both']:
            md_path = report_path.with_suffix('.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(self._format_report_as_markdown())
            log_message(f"📄 报告已保存: {md_path}")
        
        if format == 'both':
            import json
            json_path = report_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.report, f, indent=2, ensure_ascii=False)
            log_message(f"📄 JSON报告已保存: {json_path}")
        
        return report_path
    
    def _format_report_as_markdown(self):
        """将报告格式化为通俗易懂的Markdown"""
        lines = []
        
        # 标题和欢迎语
        lines.append("# 🛡️ XGBoost入侵检测系统评估报告")
        lines.append("")
        lines.append(f"> **报告生成时间：** {self.report['model_info']['generation_time']}")
        lines.append("")
        lines.append("这份报告用通俗的语言告诉你，我们的入侵检测模型表现得怎么样。")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # 1. 模型是什么
        lines.append("## 📌 1. 这个模型是什么？")
        lines.append("")
        lines.append(f"- **模型名称：** {self.report['model_info']['model_name']}")
        lines.append(f"- **模型类型：** {self.report['model_info']['model_type']}")
        lines.append(f"- **工作原理：** {self.report['model_info']['what_it_does']}")
        lines.append(f"- **使用的决策树数量：** {self.report['model_info']['tree_count']} 棵")
        lines.append("")
        lines.append("")
        
        # 2. 模型设置
        lines.append("## ⚙️ 2. 模型是如何配置的？")
        lines.append("")
        lines.append("| 设置项 | 当前值 | 这意味着什么？ |")
        lines.append("|--------|--------|----------------|")
        for param, info in self.report['parameter_configuration'].items():
            lines.append(f"| {info['simple_description']} | {info['value']} | {info['what_it_means']} |")
        lines.append("")
        
        # 3. 测试数据
        lines.append("## 📊 3. 我们用多少数据测试了模型？")
        lines.append("")
        data = self.report['data_info']
        lines.append(f"- **总共测试了 {data['total_samples']} 个网络连接**")
        lines.append(f"  - 正常连接：{data['normal_count']} 个（占 {data['normal_percentage']}）")
        lines.append(f"  - 攻击连接：{data['attack_count']} 个（占 {data['attack_percentage']}）")
        lines.append("")
        
        # 4. 模型表现如何？
        lines.append("## 🎯 4. 模型表现怎么样？")
        lines.append("")
        lines.append("| 指标 | 分数 | 评级 | 通俗解释 |")
        lines.append("|------|------|------|----------|")
        for metric, info in self.report['performance_metrics'].items():
            metric_names = {
                'accuracy': '准确率',
                'precision': '精确率',
                'recall': '召回率',
                'f1_score': '综合评分'
            }
            lines.append(f"| {metric_names.get(metric, metric)} | {info['value']} | {info['rating']} | {info['simple_explanation']} |")
        lines.append("")
        
        # 5. 详细判断结果
        lines.append("## 📋 5. 模型判断的详细结果")
        lines.append("")
        cm = self.report['confusion_matrix']
        lines.append(f"### {cm['what_is_it']}")
        lines.append("")
        lines.append("| 实际情况 | 模型判断 | 数量 | 说明 |")
        lines.append("|----------|----------|------|------|")
        results = cm['results']
        lines.append(f"| ✅ 正常 | ✅ 正常 | {results['correct_normal']['count']} | {results['correct_normal']['meaning']} |")
        lines.append(f"| ✅ 正常 | ❌ 攻击 | {results['false_alarm']['count']} | {results['false_alarm']['meaning']} |")
        lines.append(f"| ⚠️ 攻击 | ❌ 正常 | {results['missed_attack']['count']} | {results['missed_attack']['meaning']} |")
        lines.append(f"| ⚠️ 攻击 | ✅ 攻击 | {results['caught_attack']['count']} | {results['caught_attack']['meaning']} |")
        lines.append("")
        
        sec = cm['security_assessment']
        lines.append("### 🛡️ 安全评估")
        lines.append("")
        lines.append(f"- **安全等级：** {sec['level']}")
        lines.append(f"- **评估说明：** {sec['message']}")
        lines.append(f"- **误报率：** {sec['false_alarm_rate']}（正常流量被误判为攻击的比例）")
        lines.append(f"- **漏报率：** {sec['missed_rate']}（攻击流量被漏掉的比例）")
        lines.append("")
        
        # 6. 各类别表现
        if 'classification_report' in self.report:
            lines.append("## 📈 6. 对不同类型流量的识别能力")
            lines.append("")
            cr = self.report['classification_report']
            lines.append("| 流量类型 | 识别准确率 | 发现比例 | 综合评分 | 测试数量 |")
            lines.append("|----------|------------|----------|----------|----------|")
            lines.append(f"| 🟢 正常流量 | {cr['normal']['precision']} | {cr['normal']['recall']} | {cr['normal']['f1']} | {cr['normal']['sample_count']} |")
            lines.append(f"| 🔴 攻击流量 | {cr['attack']['precision']} | {cr['attack']['recall']} | {cr['attack']['f1']} | {cr['attack']['sample_count']} |")
            lines.append("")
            lines.append(f"**总体准确率：** {cr['overall_accuracy']}")
            lines.append("")
        
        # 7. AUC分数
        if 'roc_analysis' in self.report:
            lines.append("## 📊 7. 模型综合能力评分（AUC）")
            lines.append("")
            ra = self.report['roc_analysis']
            lines.append(f"> {ra.get('what_is_auc', 'AUC是衡量模型区分能力的指标')}")
            lines.append("")
            lines.append(f"- **AUC分数：** {ra['auc_score']}（最高为1）")
            lines.append(f"- **评级：** {ra['rating']}")
            lines.append(f"- **解读：** {ra['interpretation']}")
            lines.append("")
        
        # 8. 哪些特征最重要
        lines.append("## 🔍 8. 哪些网络特征最重要？")
        lines.append("")
        fi = self.report['feature_importance']
        lines.append(f"> {fi.get('explanation', '特征重要性分析')}")
        lines.append("")
        
        if fi['top_features']:
            lines.append("| 排名 | 特征名称 | 含义 | 重要性 |")
            lines.append("|------|----------|------|--------|")
            for feat in fi['top_features']:
                lines.append(f"| {feat['rank']} | `{feat['feature_name']}` | {feat['meaning']} | {feat['importance']} |")
            lines.append("")
            lines.append(f"> 💡 {fi.get('note', '')}")
        else:
            lines.append("> 无法计算特征重要性")
        lines.append("")
        
        # 9. 错误分析
        if 'error_analysis' in self.report:
            lines.append("## ❌ 9. 模型哪里判断错了？")
            lines.append("")
            ea = self.report['error_analysis']
            lines.append(f"### 总体情况")
            lines.append("")
            lines.append(f"- **错误率：** {ea['summary']['error_rate']}")
            lines.append(f"- **判断错误的连接数：** {ea['summary']['error_count']} / {ea['summary']['total_tested']}")
            lines.append("")
            lines.append("### 错误类型")
            lines.append("")
            err_types = ea['error_types']
            lines.append(f"- **误报（正常→攻击）：** {err_types['false_positives']['count']} 次 ({err_types['false_positives']['percentage']})")
            lines.append(f"  - {err_types['false_positives']['explanation']}")
            lines.append(f"- **漏报（攻击→正常）：** {err_types['false_negatives']['count']} 次 ({err_types['false_negatives']['percentage']})")
            lines.append(f"  - {err_types['false_negatives']['explanation']}")
            lines.append("")
            lines.append("### 🔧 改进建议")
            lines.append("")
            for suggestion in ea['improvement_suggestions']:
                lines.append(f"- {suggestion}")
            lines.append("")
        
        # 10. 部署建议
        lines.append("## 🚀 10. 如何使用和运维？")
        lines.append("")
        dr = self.report['deployment_recommendations']
        
        lines.append("### 需要的文件")
        lines.append("")
        for file in dr['files_needed']:
            lines.append(f"- {file}")
        lines.append("")
        
        lines.append("### 性能预期")
        lines.append("")
        lines.append(f"- **判断速度：** {dr['performance']['speed']}")
        lines.append(f"- **批量处理：** {dr['performance']['batch_speed']}")
        lines.append(f"- **内存占用：** {dr['performance']['memory']}")
        lines.append(f"- **CPU要求：** {dr['performance']['cpu']}")
        lines.append("")
        
        lines.append("### 告警建议")
        lines.append("")
        for level, action in dr['alert_suggestions'].items():
            lines.append(f"- **{level}：** {action}")
        lines.append("")
        
        lines.append("### 日常维护")
        lines.append("")
        lines.append(f"- **重新训练：** {dr['maintenance']['retrain_frequency']}")
        lines.append(f"- **监控指标：** {dr['maintenance']['monitoring']}")
        lines.append("")
        
        # 总结
        lines.append("---")
        lines.append("")
        lines.append("## 📝 总结")
        lines.append("")
        
        # 根据表现生成总结
        acc = float(self.report['performance_metrics']['accuracy']['value'].rstrip('%')) / 100
        if acc >= 0.95:
            lines.append("✅ **模型表现优秀！** 准确率超过95%，可以放心部署使用。")
            lines.append("")
            lines.append("建议：")
            lines.append("1. 可以直接部署到生产环境")
            lines.append("2. 建议定期（每月）用新数据重新训练，保持模型时效性")
            lines.append("3. 关注漏报率变化，这是安全最关键指标")
        elif acc >= 0.9:
            lines.append("👍 **模型表现良好！** 准确率超过90%，基本满足使用需求。")
            lines.append("")
            lines.append("建议：")
            lines.append("1. 可以先小范围试用，观察实际效果")
            lines.append("2. 重点关注误报情况，避免干扰运维人员")
            lines.append("3. 收集更多攻击样本，进一步提升模型")
        elif acc >= 0.8:
            lines.append("⚠️ **模型表现及格**，但还有提升空间。")
            lines.append("")
            lines.append("建议：")
            lines.append("1. 建议配合人工审核使用")
            lines.append("2. 收集更多高质量训练数据")
            lines.append("3. 考虑调整模型参数或尝试其他算法")
        else:
            lines.append("🔴 **模型表现需要改进**，暂不建议直接部署。")
            lines.append("")
            lines.append("建议：")
            lines.append("1. 检查数据质量，确保训练数据准确标注")
            lines.append("2. 增加训练数据量，特别是攻击样本")
            lines.append("3. 尝试特征工程，提取更有区分度的特征")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*本报告由 XGBoost入侵检测系统自动生成*")
        
        return "\n".join(lines)


def generate_model_report():
    """生成完整的模型报告"""
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
        report = generator.generate_complete_report(evaluation_results)
        
        report_path = generator.save_report(format='md')
        
        log_message(f"\n✅ 报告生成完成！")
        log_message(f"📄 报告位置: {report_path}")
        
        return report
        
    except Exception as e:
        log_message(f"报告生成失败: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    generate_model_report()