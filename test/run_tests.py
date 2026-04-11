"""
测试程序5：一键运行所有测试
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

import subprocess

def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("Starting Complete Test Suite")
    print("="*70)
    
    # 获取test目录
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    tests = [
        ("Basic Functionality Test", "test_model.py"),
        ("Advanced Performance Test", "test_advanced.py"),
        ("Visualization Test", "test_visual.py")
    ]
    
    results = []
    
    for test_name, test_file in tests:
        print(f"\n{'='*70}")
        print(f"Running: {test_name}")
        print(f"{'='*70}")
        
        test_path = os.path.join(test_dir, test_file)
        
        if not os.path.exists(test_path):
            print(f"[FAIL] Test file not found: {test_path}")
            results.append({'测试': test_name, '状态': '文件缺失'})
            continue
        
        try:
            # 使用更简单的执行方式，避免编码问题
            print(f"Executing: python {test_file}")
            
            # 直接在当前进程中执行，而不是创建子进程
            # 这样可以避免编码问题
            import importlib.util
            import runpy
            
            try:
                # 尝试使用runpy执行
                runpy.run_path(test_path, run_name="__main__")
                print(f"\n[SUCCESS] {test_name} completed")
                results.append({'测试': test_name, '状态': '通过'})
            except Exception as e:
                print(f"\n[FAIL] {test_name} failed: {e}")
                results.append({'测试': test_name, '状态': '失败'})
                
        except Exception as e:
            print(f"\n[ERROR] {test_name} error: {e}")
            results.append({'测试': test_name, '状态': '错误'})
    
    # 打印总结
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    try:
        import pandas as pd
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        passed = sum(results_df['状态'] == '通过')
        total = len(results_df)
        
        print(f"\nPass Rate: {passed}/{total} ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("\n[SUCCESS] All tests passed! Models are ready to use!")
        else:
            print("\n[WARNING] Some tests failed, please check the issues")
    except:
        for r in results:
            print(f"  {r['测试']}: {r['状态']}")

if __name__ == "__main__":
    run_all_tests()