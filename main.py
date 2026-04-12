"""
主入口脚本 - XGBoost入侵检测系统（简化版）
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 模块配置
MODULES = {
    '1': ('数据探索', 'data_exploration', False),
    '2': ('数据预处理', 'preprocess', False),
    '3': ('训练模型', 'train_models', False),
    '4': ('评估模型', 'evaluate', False),
    '5': ('超参数调优', 'hyperparameter_tuning', True),
    '6': ('生成报告', 'report_generator', False),
}


def run_module(choice):
    """执行模块"""
    name, module_name, need_confirm = MODULES[choice]
    
    # 超参数调优需要确认
    if need_confirm:
        print("⚠️  注意: 这将花费15-30分钟")
        if input("确认继续？(y/n): ").strip().lower() != 'y':
            print("已取消")
            return
    
    print(f"\n{'='*60}\n启动{name}\n{'='*60}")
    
    # 动态导入并执行
    module = __import__(f'src.{module_name}', fromlist=['main'])
    
    # 特殊处理：不同模块的入口函数不同
    if choice == '6':
        module.generate_model_report()
    else:
        module.main()


def run_pipeline():
    """运行完整流程"""
    print("\n🔄 开始运行完整流程...\n")
    
    steps = ['1', '2', '3', '4', '6']
    for step in steps:
        name, module_name, _ = MODULES[step]
        print(f"\n{'='*60}\n>>> 步骤: {name}\n{'='*60}")
        
        module = __import__(f'src.{module_name}', fromlist=['main'])
        
        # 根据步骤执行不同函数
        if step == '2':
            if module.main() is None:
                print("❌ 数据预处理失败，停止流程")
                return
        elif step == '3':
            _, model = module.train_xgboost_model()
            if model is None:
                print("❌ 模型训练失败，停止流程")
                return
        elif step == '4':
            if module.evaluate_xgboost_model() is None:
                print("❌ 模型评估失败，停止流程")
                return
        elif step == '6':
            module.generate_model_report()
        else:
            module.main()
    
    print(f"\n{'='*60}\n✅ 完整流程运行完成！\n{'='*60}")


def main():
    """主函数"""
    print("\n🎯 XGBoost入侵检测系统模型训练- UNSW-NB15")
    
    while True:
        # 简洁菜单
        print("\n" + "="*50)
        print("1.数据探索 2.预处理 3.训练 4.评估 5.调优 6.报告 7.完整流程 0.退出")
        print("="*50)
        
        choice = input("请选择 [0-7]: ").strip()
        
        if choice == '0':
            print("\n👋 再见！")
            break
        elif choice == '7':
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