"""
主入口脚本 - 运行完整的训练流程
"""

import sys
import os

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_menu():
    """打印菜单"""
    print("\n" + "="*60)
    print("UNSW-NB15 入侵检测系统")
    print("="*60)
    print("1. 数据探索 (EDA)")
    print("2. 数据预处理")
    print("3. 训练模型")
    print("4. 评估模型")
    print("5. 超参数调优")
    print("6. 运行完整流程 (1->2->3->4)")
    print("0. 退出")
    print("="*60)

def run_complete_pipeline():
    """运行完整流程"""
    print("\n开始运行完整流程...\n")
    
    # 1. 数据探索
    print("\n>>> 步骤1: 数据探索")
    from src import data_exploration as de
    de.main()
    
    # 2. 数据预处理
    print("\n>>> 步骤2: 数据预处理")
    from src import preprocess as pp
    pp.main()
    
    # 3. 训练模型
    print("\n>>> 步骤3: 训练模型")
    from src import train_models as tm
    tm.train_models()
    
    # 4. 评估模型
    print("\n>>> 步骤4: 评估模型")
    from src import evaluate as ev
    ev.evaluate_best_model()
    
    print("\n完整流程运行完成！")

def main():
    while True:
        print_menu()
        choice = input("请输入选择: ").strip()
        
        if choice == '1':
            from src import data_exploration as de
            de.main()
        elif choice == '2':
            from src import preprocess as pp
            pp.main()
        elif choice == '3':
            from src import train_models as tm
            tm.train_models()
        elif choice == '4':
            from src import evaluate as ev
            ev.evaluate_best_model()
        elif choice == '5':
            from src import hyperparameter_tuning as ht
            ht.main()
        elif choice == '6':
            run_complete_pipeline()
        elif choice == '0':
            print("退出程序")
            break
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()