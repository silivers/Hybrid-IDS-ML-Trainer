"""
主入口脚本 - XGBoost入侵检测系统（简化版）
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 模块配置 - 统一为2个值
MODULES = {
    '1': ('数据探索', 'data_exploration'),
    '2': ('数据预处理', 'preprocess'),
    '3': ('训练模型', 'train_models'),
    '4': ('评估模型', 'evaluate'),
    '5': ('生成报告', 'report_generator'),
}


def run_module(choice):
    """执行模块"""
    name, module_name = MODULES[choice]  # 修复：去掉末尾的逗号
    
    print(f"\n{'='*60}\n启动{name}\n{'='*60}")
    
    # 动态导入并执行
    module = __import__(f'src.{module_name}', fromlist=['main'])
    
    # 特殊处理：不同模块的入口函数不同
    if choice == '5':  # 修复：应该是5，不是6
        if hasattr(module, 'generate_model_report'):
            module.generate_model_report()
        else:
            print(f"❌ 模块 {module_name} 没有 generate_model_report 函数")
    elif choice == '4':  # 评估模块
        if hasattr(module, 'evaluate_xgboost_model'):
            module.evaluate_xgboost_model()
        elif hasattr(module, 'main'):
            module.main()
        else:
            print(f"❌ 模块 {module_name} 没有可用的入口函数")
    elif choice == '3':  # 训练模块
        if hasattr(module, 'train_xgboost_model'):
            module.train_xgboost_model()
        elif hasattr(module, 'main'):
            module.main()
        else:
            print(f"❌ 模块 {module_name} 没有可用的入口函数")
    else:  # 数据探索和预处理
        if hasattr(module, 'main'):
            module.main()
        else:
            print(f"❌ 模块 {module_name} 没有 main 函数")


def run_pipeline():
    """运行完整流程"""
    print("\n🔄 开始运行完整流程...\n")
    
    steps = ['1', '2', '3', '4', '5']
    for step in steps:
        name, module_name = MODULES[step]  # 修复：解包2个值，去掉 _
        print(f"\n{'='*60}\n>>> 步骤: {name}\n{'='*60}")
        
        module = __import__(f'src.{module_name}', fromlist=['main'])
        
        # 根据步骤执行不同函数
        if step == '2':  # 预处理
            if hasattr(module, 'main'):
                result = module.main()
                if result is None:
                    print("❌ 数据预处理失败，停止流程")
                    return
            else:
                print(f"❌ 模块 {module_name} 没有 main 函数")
                return
        elif step == '3':  # 训练
            if hasattr(module, 'train_xgboost_model'):
                result = module.train_xgboost_model()
                # 处理返回值可能是元组的情况
                if result is None:
                    print("❌ 模型训练失败，停止流程")
                    return
                elif isinstance(result, tuple) and len(result) >= 2 and result[1] is None:
                    print("❌ 模型训练失败，停止流程")
                    return
            else:
                print(f"❌ 模块 {module_name} 没有 train_xgboost_model 函数")
                return
        elif step == '4':  # 评估
            if hasattr(module, 'evaluate_xgboost_model'):
                if module.evaluate_xgboost_model() is None:
                    print("❌ 模型评估失败，停止流程")
                    return
            else:
                print(f"❌ 模块 {module_name} 没有 evaluate_xgboost_model 函数")
                return
        elif step == '5':  # 报告
            if hasattr(module, 'generate_model_report'):
                module.generate_model_report()
            elif hasattr(module, 'main'):
                module.main()
            else:
                print(f"❌ 模块 {module_name} 没有可用的入口函数")
                return
        else:  # 数据探索
            if hasattr(module, 'main'):
                module.main()
            else:
                print(f"❌ 模块 {module_name} 没有 main 函数")
                return
    
    print(f"\n{'='*60}\n✅ 完整流程运行完成！\n{'='*60}")


def main():
    """主函数"""
    print("\n🎯 XGBoost入侵检测系统模型训练 - UNSW-NB15")
    
    while True:
        # 简洁菜单
        print("\n" + "="*50)
        print("1.数据探索 2.预处理 3.训练 4.评估 5.报告 6.完整流程 0.退出")
        print("="*50)
        
        choice = input("请选择 [0-6]: ").strip()
        
        if choice == '0':
            print("\n👋 再见！")
            break
        elif choice == '6':
            run_pipeline()
        elif choice in MODULES:
            run_module(choice)
        else:
            print("❌ 无效选择，请重新输入")
        
        if choice != '0':
            input("\n按Enter键继续...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被中断，再见！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()