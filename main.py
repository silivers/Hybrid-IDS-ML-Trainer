"""
主入口脚本 - XGBoost入侵检测系统
"""

import sys
import os

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_menu():
    """打印菜单"""
    print("\n" + "="*60)
    print("🔒 XGBoost入侵检测系统 - UNSW-NB15")
    print("="*60)
    print("1. 📊 数据探索 (EDA)")
    print("2. 🔧 数据预处理")
    print("3. 🚀 训练XGBoost模型")
    print("4. 📈 评估XGBoost模型")
    print("5. ⚡ 超参数调优 (优化XGBoost)")
    print("6. 📄 生成完整模型报告")
    print("7. 🔄 运行完整流程 (1->2->3->4->6)")
    print("0. ❌ 退出")
    print("="*60)

def run_complete_pipeline():
    """运行完整流程"""
    print("\n🔄 开始运行完整流程...\n")
    
    # 1. 数据探索
    print("\n" + "="*60)
    print(">>> 步骤1: 数据探索")
    print("="*60)
    from src import data_exploration as de
    de.main()
    
    # 2. 数据预处理
    print("\n" + "="*60)
    print(">>> 步骤2: 数据预处理")
    print("="*60)
    from src import preprocess as pp
    result = pp.main()
    if result is None:
        print("❌ 数据预处理失败，停止流程")
        return
    
    # 3. 训练XGBoost模型
    print("\n" + "="*60)
    print(">>> 步骤3: 训练XGBoost模型")
    print("="*60)
    from src import train_models as tm
    metrics, model = tm.train_xgboost_model()
    if model is None:
        print("❌ 模型训练失败，停止流程")
        return
    
    # 4. 评估模型
    print("\n" + "="*60)
    print(">>> 步骤4: 评估XGBoost模型")
    print("="*60)
    from src import evaluate as ev
    eval_results = ev.evaluate_xgboost_model()
    if eval_results is None:
        print("❌ 模型评估失败，停止流程")
        return
    
    # 5. 生成完整报告
    print("\n" + "="*60)
    print(">>> 步骤5: 生成完整模型报告")
    print("="*60)
    from src import report_generator as rg
    rg.generate_model_report()
    
    print("\n" + "="*60)
    print("✅ 完整流程运行完成！")
    print("="*60)

def main():
    """主函数"""
    print("\n🎯 欢迎使用XGBoost入侵检测系统")
    print("基于UNSW-NB15数据集的网络攻击检测")
    
    while True:
        print_menu()
        choice = input("请输入选择 [0-7]: ").strip()
        
        if choice == '1':
            print("\n" + "="*60)
            print("启动数据探索模块")
            print("="*60)
            from src import data_exploration as de
            de.main()
            
        elif choice == '2':
            print("\n" + "="*60)
            print("启动数据预处理模块")
            print("="*60)
            from src import preprocess as pp
            pp.main()
            
        elif choice == '3':
            print("\n" + "="*60)
            print("启动XGBoost模型训练")
            print("="*60)
            from src import train_models as tm
            tm.main()
            
        elif choice == '4':
            print("\n" + "="*60)
            print("启动模型评估模块")
            print("="*60)
            from src import evaluate as ev
            ev.main()
            
        elif choice == '5':
            print("\n" + "="*60)
            print("启动超参数调优模块")
            print("="*60)
            print("⚠️  注意: 这将花费15-30分钟")
            confirm = input("确认继续？(y/n): ").strip().lower()
            if confirm == 'y':
                from src import hyperparameter_tuning as ht
                ht.main()
            else:
                print("已取消")
        
        elif choice == '6':
            print("\n" + "="*60)
            print("生成完整模型报告")
            print("="*60)
            from src import report_generator as rg
            rg.generate_model_report()
            
        elif choice == '7':
            run_complete_pipeline()
            
        elif choice == '0':
            print("\n👋 感谢使用XGBoost入侵检测系统！")
            print("再见！")
            break
            
        else:
            print("❌ 无效选择，请重新输入 (0-7)")
        
        # 询问是否继续
        if choice != '0':
            print("\n" + "-"*60)
            input("按Enter键返回主菜单...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断，再见！")
    except Exception as e:
        print(f"\n❌ 程序出错: {e}")
        import traceback
        traceback.print_exc()