"""
测试程序1：基础模型测试
测试模型是否能正常加载和预测
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

class BasicModelTest:
    """基础模型测试类"""
    
    def __init__(self):
        # 获取项目根目录
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.project_root, "models")
        self.test_results = []
        
    def test_model_loading(self):
        """测试1：模型加载"""
        print("\n" + "="*60)
        print("测试1：检查模型文件")
        print("="*60)
        
        model_files = [
            ('xgboost.pkl', 'XGBoost'),
            ('random_forest.pkl', '随机森林'), 
            ('gradient_boosting.pkl', '梯度提升'),
            ('logistic_regression.pkl', '逻辑回归'),
            ('scaler.pkl', '标准化器'),
            ('label_encoders.pkl', '标签编码器')
        ]
        
        for model_file, model_name in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    print(f"[OK] {model_name}: 加载成功 ({type(model).__name__})")
                    self.test_results.append({'测试': f'加载{model_name}', '状态': '通过'})
                except Exception as e:
                    print(f"[FAIL] {model_name}: 加载失败 - {e}")
                    self.test_results.append({'测试': f'加载{model_name}', '状态': '失败'})
            else:
                print(f"[WARN] {model_name}: 文件不存在")
                self.test_results.append({'测试': f'加载{model_name}', '状态': '文件缺失'})
    
    def test_prediction(self):
        """测试2：预测功能"""
        print("\n" + "="*60)
        print("测试2：模型预测功能")
        print("="*60)
        
        # 加载最佳模型
        model_path = os.path.join(self.models_dir, 'xgboost.pkl')
        if not os.path.exists(model_path):
            print("[FAIL] xgboost.pkl 不存在，跳过预测测试")
            return
        
        model = joblib.load(model_path)
        
        # 获取特征数量（从scaler或模型中推断）
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            n_features = scaler.mean_.shape[0] if hasattr(scaler, 'mean_') else 43
        else:
            n_features = 43  # UNSW-NB15 默认特征数
        
        # 创建测试数据
        test_data = np.random.rand(1, n_features)
        print(f"   测试数据形状: {test_data.shape}")
        
        try:
            # 测试预测
            prediction = model.predict(test_data)
            result_text = "攻击" if prediction[0] == 1 else "正常"
            print(f"[OK] 预测成功: {prediction[0]} ({result_text})")
            
            # 测试概率预测
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(test_data)
                print(f"[OK] 概率预测成功: 正常={probability[0][0]:.4f}, 攻击={probability[0][1]:.4f}")
            
            self.test_results.append({'测试': '预测功能', '状态': '通过'})
        except Exception as e:
            print(f"[FAIL] 预测失败: {e}")
            self.test_results.append({'测试': '预测功能', '状态': '失败'})
    
    def test_batch_prediction(self):
        """测试3：批量预测"""
        print("\n" + "="*60)
        print("测试3：批量预测性能")
        print("="*60)
        
        model_path = os.path.join(self.models_dir, 'xgboost.pkl')
        if not os.path.exists(model_path):
            print("[FAIL] 模型不存在，跳过批量测试")
            return
        
        model = joblib.load(model_path)
        
        # 获取特征数量
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            n_features = scaler.mean_.shape[0] if hasattr(scaler, 'mean_') else 43
        else:
            n_features = 43
        
        # 测试不同批量大小
        batch_sizes = [10, 100, 1000]
        
        for batch_size in batch_sizes:
            test_data = np.random.rand(batch_size, n_features)
            
            import time
            start_time = time.time()
            predictions = model.predict(test_data)
            elapsed = time.time() - start_time
            
            print(f"   批量大小 {batch_size:5}: {elapsed:.4f}秒 ({batch_size/elapsed:.0f} 样本/秒)")
        
        self.test_results.append({'测试': '批量预测', '状态': '通过'})
    
    def test_data_compatibility(self):
        """测试4：数据兼容性"""
        print("\n" + "="*60)
        print("测试4：数据格式兼容性")
        print("="*60)
        
        model_path = os.path.join(self.models_dir, 'xgboost.pkl')
        if not os.path.exists(model_path):
            print("[FAIL] 模型不存在，跳过兼容性测试")
            return
        
        model = joblib.load(model_path)
        
        # 获取特征数量
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            n_features = scaler.mean_.shape[0] if hasattr(scaler, 'mean_') else 43
        else:
            n_features = 43
        
        # 测试不同数据格式
        test_cases = [
            ("numpy数组", np.random.rand(1, n_features)),
            ("列表", [np.random.rand(n_features).tolist()]),
            ("二维列表", [[np.random.rand() for _ in range(n_features)]])
        ]
        
        for data_name, test_data in test_cases:
            try:
                prediction = model.predict(test_data)
                print(f"[OK] {data_name}: 兼容")
            except Exception as e:
                print(f"[FAIL] {data_name}: 不兼容 - {e}")
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "="*60)
        print("测试总结")
        print("="*60)
        
        if self.test_results:
            results_df = pd.DataFrame(self.test_results)
            print(results_df.to_string(index=False))
            
            passed = sum(results_df['状态'] == '通过')
            total = len(results_df)
            
            print(f"\n通过率: {passed}/{total} ({passed/total*100:.1f}%)")
            
            if passed == total:
                print("\n[SUCCESS] 所有测试通过！模型可以正常使用！")
            else:
                print("\n[WARNING] 部分测试失败，请检查问题")
        else:
            print("没有运行任何测试")

def main():
    print("\n开始基础模型测试")
    print(f"项目目录: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
    
    tester = BasicModelTest()
    tester.test_model_loading()
    tester.test_prediction()
    tester.test_batch_prediction()
    tester.test_data_compatibility()
    tester.print_summary()

if __name__ == "__main__":
    main()