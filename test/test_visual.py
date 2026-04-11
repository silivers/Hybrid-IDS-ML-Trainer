"""
测试程序3：可视化测试
生成图表展示模型性能
"""

import sys
import os
# 设置控制台编码为UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import matplotlib
# 使用非交互式后端，避免弹出窗口
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class VisualModelTest:
    """可视化测试类"""
    
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.project_root, "models")
        self.output_dir = os.path.join(self.project_root, "results", "figures")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_test_data(self):
        """加载测试数据"""
        try:
            # 尝试使用预处理模块
            from src.preprocess import get_data
            X_train, X_test, y_train, y_test = get_data()
            return X_test, y_test
        except Exception as e:
            print(f"[WARN] 使用预处理模块失败: {e}")
            
            # 直接加载原始数据并预处理
            try:
                from src.preprocess import preprocess_and_cache
                result = preprocess_and_cache()
                if result:
                    X_train, X_test, y_train, y_test = result
                    return X_test, y_test
            except Exception as e2:
                print(f"[FAIL] 无法加载测试数据: {e2}")
                return None, None
    
    def plot_confusion_matrix(self, model_name='xgboost'):
        """绘制混淆矩阵"""
        print(f"\n绘制 {model_name} 混淆矩阵...")
        
        # 加载模型和数据
        model_path = os.path.join(self.models_dir, f'{model_name}.pkl')
        if not os.path.exists(model_path):
            print(f"[FAIL] 模型不存在: {model_path}")
            return
        
        model = joblib.load(model_path)
        X_test, y_test = self.load_test_data()
        
        if X_test is None:
            return
        
        # 预测
        print("   正在进行预测...")
        y_pred = model.predict(X_test)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 绘制
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f'{model_name.upper()} Confusion Matrix', fontsize=14)
        plt.colorbar()
        
        classes = ['Normal', 'Attack']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # 添加数值
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.output_dir, f'{model_name}_confusion_matrix.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] 已保存: {save_path}")
        plt.close()
        
        # 打印数值
        tn, fp, fn, tp = cm.ravel()
        print(f"\n   混淆矩阵数值:")
        print(f"     True Negatives (Normal->Normal): {tn}")
        print(f"     False Positives (Normal->Attack): {fp}")
        print(f"     False Negatives (Attack->Normal): {fn}")
        print(f"     True Positives (Attack->Attack): {tp}")
    
    def plot_roc_curve(self, model_name='xgboost'):
        """绘制ROC曲线"""
        print(f"\n绘制 {model_name} ROC曲线...")
        
        model_path = os.path.join(self.models_dir, f'{model_name}.pkl')
        if not os.path.exists(model_path):
            print(f"[FAIL] 模型不存在: {model_path}")
            return
        
        model = joblib.load(model_path)
        X_test, y_test = self.load_test_data()
        
        if X_test is None or not hasattr(model, 'predict_proba'):
            print("[WARN] 模型不支持概率预测")
            return
        
        # 预测概率
        print("   正在计算概率...")
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # 计算ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # 绘制
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'{model_name.upper()} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{model_name.upper()} ROC Curve', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # 保存
        save_path = os.path.join(self.output_dir, f'{model_name}_roc_curve.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] 已保存: {save_path}")
        print(f"   AUC Score: {roc_auc:.4f}")
        plt.close()
    
    def plot_model_comparison(self):
        """绘制模型对比图"""
        print("\n绘制模型性能对比图...")
        
        # 读取训练结果
        results_path = os.path.join(self.models_dir, 'training_results.csv')
        if not os.path.exists(results_path):
            print("[FAIL] 训练结果文件不存在")
            return
        
        results = pd.read_csv(results_path)
        
        # 创建对比图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 子图1：F1分数对比
        models = results['Model']
        f1_scores = results['F1-Score']
        
        colors = ['#2ecc71' if x == max(f1_scores) else '#3498db' for x in f1_scores]
        axes[0].barh(models, f1_scores, color=colors)
        axes[0].set_xlabel('F1 Score', fontsize=12)
        axes[0].set_title('Model F1 Score Comparison', fontsize=14)
        axes[0].set_xlim(0, 1)
        
        # 添加数值标签
        for i, (model, score) in enumerate(zip(models, f1_scores)):
            axes[0].text(score + 0.01, i, f'{score:.4f}', va='center')
        
        # 子图2：训练时间对比（如果有）
        if 'TrainTime' in results.columns:
            train_times = results['TrainTime']
            axes[1].bar(models, train_times, color='#e74c3c')
            axes[1].set_ylabel('Training Time (seconds)', fontsize=12)
            axes[1].set_title('Model Training Time Comparison', fontsize=14)
            axes[1].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for i, (model, time) in enumerate(zip(models, train_times)):
                axes[1].text(i, time + 0.5, f'{time:.1f}s', ha='center')
        else:
            axes[1].text(0.5, 0.5, 'No training time data', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Training Time', fontsize=14)
        
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] 已保存: {save_path}")
        plt.close()
    
    def plot_performance_radar(self):
        """绘制性能雷达图"""
        print("\n绘制性能雷达图...")
        
        results_path = os.path.join(self.models_dir, 'training_results.csv')
        if not os.path.exists(results_path):
            print("[FAIL] 训练结果文件不存在")
            return
        
        results = pd.read_csv(results_path)
        
        # 准备数据
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        # 只保留存在的指标
        metrics = [m for m in metrics if m in results.columns]
        
        if len(metrics) < 3:
            print("[WARN] 指标数据不足，跳过雷达图")
            return
        
        models = results['Model'].tolist()
        
        # 创建雷达图
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, model in enumerate(models):
            values = [results.loc[idx, m] for m in metrics]
            values += values[:1]  # 闭合
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', size=15, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        # 保存
        save_path = os.path.join(self.output_dir, 'performance_radar.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] 已保存: {save_path}")
        plt.close()

def main():
    print("\n开始可视化测试")
    print(f"项目目录: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
    
    tester = VisualModelTest()
    
    # 生成各种图表
    tester.plot_confusion_matrix('xgboost')
    tester.plot_roc_curve('xgboost')
    tester.plot_model_comparison()
    tester.plot_performance_radar()
    
    print(f"\n[SUCCESS] 所有图表已保存到: {tester.output_dir}")

if __name__ == "__main__":
    main()