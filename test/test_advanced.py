"""
测试程序2：高级模型测试
使用真实测试数据验证模型性能
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

class AdvancedModelTest:
    """高级模型测试类"""
    
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.project_root, "models")
        self.test_data_path = os.path.join(
            self.project_root, 
            "dataset", 
            "Training and Testing Sets", 
            "UNSW_NB15_testing-set.csv"
        )
        
    def load_test_data(self):
        """加载测试数据"""
        print("\n" + "="*60)
        print("加载测试数据集")
        print("="*60)
        
        if not os.path.exists(self.test_data_path):
            print(f"[FAIL] 测试数据不存在: {self.test_data_path}")
            return None
        
        df = pd.read_csv(self.test_data_path)
        print(f"[OK] 测试数据加载成功")
        print(f"   数据形状: {df.shape}")
        print(f"   列数: {len(df.columns)}")
        print(f"   列名: {df.columns.tolist()[:10]}...")
        
        # 显示标签分布
        if 'label' in df.columns:
            normal = (df['label'] == 0).sum()
            attack = (df['label'] == 1).sum()
            print(f"   正常样本: {normal} ({normal/len(df)*100:.1f}%)")
            print(f"   攻击样本: {attack} ({attack/len(df)*100:.1f}%)")
        
        return df
    
    def preprocess_test_data(self, df):
        """预处理测试数据"""
        print("\n" + "="*60)
        print("预处理测试数据")
        print("="*60)
        
        # 加载预处理工具
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        encoder_path = os.path.join(self.models_dir, 'label_encoders.pkl')
        
        if not os.path.exists(scaler_path):
            print("[FAIL] 标准化器不存在，请先运行预处理")
            return None, None
        
        scaler = joblib.load(scaler_path)
        encoders = joblib.load(encoder_path) if os.path.exists(encoder_path) else {}
        
        # 复制数据
        df_processed = df.copy()
        
        # 删除不必要的列（与训练时保持一致）
        cols_to_drop = ['id']  # id列不需要
        for col in cols_to_drop:
            if col in df_processed.columns:
                df_processed = df_processed.drop(columns=[col])
                print(f"   删除列: {col}")
        
        # 处理类别特征
        categorical_cols = ['proto', 'service', 'state']
        
        for col in categorical_cols:
            if col in df_processed.columns and col in encoders:
                print(f"   处理类别列: {col}")
                df_processed[col] = df_processed[col].astype(str)
                # 处理未知类别
                known_classes = set(encoders[col].classes_)
                df_processed[col] = df_processed[col].apply(
                    lambda x: encoders[col].transform([x])[0] if x in known_classes else -1
                )
        
        # 处理缺失值
        if df_processed.isnull().sum().sum() > 0:
            print("   填充缺失值...")
            df_processed = df_processed.fillna(0)
        
        # 分离特征和标签
        if 'label' in df_processed.columns:
            y = df_processed['label'].values
            # 删除标签列和攻击类别列
            cols_to_drop_for_X = ['label']
            if 'attack_cat' in df_processed.columns:
                cols_to_drop_for_X.append('attack_cat')
            X = df_processed.drop(columns=cols_to_drop_for_X)
        else:
            X = df_processed
            y = None
        
        print(f"   特征列数: {X.shape[1]}")
        print(f"   特征列名: {X.columns.tolist()[:5]}...")
        
        # 获取训练时的特征名（如果可用）
        feature_names_path = os.path.join(self.models_dir, 'feature_columns.npy')
        if os.path.exists(feature_names_path):
            train_features = np.load(feature_names_path, allow_pickle=True).tolist()
            print(f"   训练时特征数: {len(train_features)}")
            
            # 确保特征列一致
            # 添加缺失的特征列
            for col in train_features:
                if col not in X.columns:
                    X[col] = 0
                    print(f"   添加缺失特征: {col}")
            
            # 只保留训练时使用的特征
            X = X[train_features]
            print(f"   调整后特征数: {X.shape[1]}")
        
        # 标准化
        X_scaled = scaler.transform(X)
        
        print(f"[OK] 预处理完成")
        print(f"   特征形状: {X_scaled.shape}")
        print(f"   特征数量: {X_scaled.shape[1]}")
        
        return X_scaled, y
    
    def test_all_models(self, X_test, y_test):
        """测试所有模型"""
        print("\n" + "="*60)
        print("测试所有模型性能")
        print("="*60)
        
        model_files = {
            'xgboost.pkl': 'XGBoost',
            'random_forest.pkl': '随机森林',
            'gradient_boosting.pkl': '梯度提升',
            'logistic_regression.pkl': '逻辑回归'
        }
        
        results = []
        
        for model_file, model_name in model_files.items():
            model_path = os.path.join(self.models_dir, model_file)
            if not os.path.exists(model_path):
                print(f"[WARN] {model_name} 模型不存在，跳过")
                continue
            
            print(f"\n测试 {model_name}...")
            model = joblib.load(model_path)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 计算指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            print(f"   准确率: {accuracy:.4f}")
            print(f"   精确率: {precision:.4f}")
            print(f"   召回率: {recall:.4f}")
            print(f"   F1分数: {f1:.4f}")
            
            # AUC
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
                print(f"   AUC: {auc:.4f}")
            else:
                auc = None
            
            results.append({
                '模型': model_name,
                '准确率': f"{accuracy:.4f}",
                '精确率': f"{precision:.4f}",
                '召回率': f"{recall:.4f}",
                'F1分数': f"{f1:.4f}",
                'AUC': f"{auc:.4f}" if auc else "N/A"
            })
        
        # 显示结果表格
        if results:
            print("\n" + "="*60)
            print("模型性能对比")
            print("="*60)
            results_df = pd.DataFrame(results)
            print(results_df.to_string(index=False))
            
            # 保存结果
            output_dir = os.path.join(self.project_root, "results", "reports")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "model_comparison.csv")
            results_df.to_csv(output_path, index=False)
            print(f"\n[SAVED] 结果已保存: {output_path}")
        
        return results
    
    def test_attack_types(self, df, X_test, y_test):
        """测试不同攻击类型的检测能力"""
        print("\n" + "="*60)
        print("不同攻击类型检测能力测试")
        print("="*60)
        
        if 'attack_cat' not in df.columns:
            print("[WARN] 数据中没有攻击类型信息")
            return None
        
        model_path = os.path.join(self.models_dir, 'xgboost.pkl')
        if not os.path.exists(model_path):
            print("[FAIL] 模型不存在")
            return None
        
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        
        # 只分析攻击样本
        attack_mask = y_test == 1
        if not np.any(attack_mask):
            print("[WARN] 测试集中没有攻击样本")
            return None
        
        # 获取对应的攻击类型
        attack_types_df = df[attack_mask]['attack_cat'].values
        attack_pred = y_pred[attack_mask]
        
        # 统计每种攻击的检测率
        attack_results = {}
        
        for attack_type in set(attack_types_df):
            type_mask = attack_types_df == attack_type
            total = type_mask.sum()
            correct = (attack_pred[type_mask] == 1).sum()
            detection_rate = correct / total if total > 0 else 0
            
            attack_results[attack_type] = {
                '总数': total,
                '检测数': correct,
                '检测率': f"{detection_rate:.4f}",
                '检测率%': f"{detection_rate*100:.1f}%"
            }
        
        # 显示结果
        results_df = pd.DataFrame(attack_results).T
        results_df = results_df.sort_values('检测率', ascending=False)
        
        print("\n攻击类型检测率排名:")
        print(results_df.to_string())
        
        return results_df
    
    def test_error_cases(self, X_test, y_test):
        """分析错误预测案例"""
        print("\n" + "="*60)
        print("错误案例分析")
        print("="*60)
        
        model_path = os.path.join(self.models_dir, 'xgboost.pkl')
        if not os.path.exists(model_path):
            print("[FAIL] 模型不存在")
            return
        
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        
        # 找出错误预测
        errors = np.where(y_pred != y_test)[0]
        
        if len(errors) == 0:
            print("[SUCCESS] 完美！没有错误预测！")
            return
        
        print(f"总错误数: {len(errors)}/{len(y_test)} ({len(errors)/len(y_test)*100:.2f}%)")
        
        # 错误类型统计
        false_positives = np.where((y_pred == 1) & (y_test == 0))[0]
        false_negatives = np.where((y_pred == 0) & (y_test == 1))[0]
        
        print(f"\n错误类型:")
        print(f"  误报 (正常->攻击): {len(false_positives)} 个 ({len(false_positives)/len(errors)*100:.1f}%)")
        print(f"  漏报 (攻击->正常): {len(false_negatives)} 个 ({len(false_negatives)/len(errors)*100:.1f}%)")
        
        # 计算误报率和漏报率
        normal_count = np.sum(y_test == 0)
        attack_count = np.sum(y_test == 1)
        
        if normal_count > 0:
            fp_rate = len(false_positives) / normal_count
            print(f"\n误报率: {fp_rate:.4f} ({fp_rate*100:.2f}%)")
        
        if attack_count > 0:
            fn_rate = len(false_negatives) / attack_count
            print(f"漏报率: {fn_rate:.4f} ({fn_rate*100:.2f}%)")

def main():
    print("\n开始高级模型测试")
    print(f"项目目录: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
    
    tester = AdvancedModelTest()
    
    # 1. 加载测试数据
    df = tester.load_test_data()
    if df is None:
        return
    
    # 2. 预处理
    X_test, y_test = tester.preprocess_test_data(df)
    if X_test is None:
        return
    
    # 3. 测试所有模型
    tester.test_all_models(X_test, y_test)
    
    # 4. 测试攻击类型检测
    tester.test_attack_types(df, X_test, y_test)
    
    # 5. 错误案例分析
    tester.test_error_cases(X_test, y_test)
    
    print("\n[SUCCESS] 高级测试完成！")

if __name__ == "__main__":
    main()