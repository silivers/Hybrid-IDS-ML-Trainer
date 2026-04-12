# XGBoost入侵检测系统



## 项目概述
基于XGBoost算法的网络入侵检测系统模型训练，使用UNSW-NB15数据集进行训练和评估。本项目提供了完整的数据预处理、模型训练、超参数调优、评估和报告生成功能。

### 核心功能

- 数据探索：分析数据集结构、标签分布和攻击类型
- 数据预处理：处理类别特征、缺失值填充、特征标准化
- 模型训练：使用XGBoost算法训练入侵检测模型
- 超参数调优：网格搜索优化模型参数
- 模型评估：计算准确率、精确率、召回率、F1分数等指标
- 报告生成：自动生成通俗易懂的Markdown格式评估报告

## 项目结构

├── dataset/
│   └── Training and Testing Sets/
│       ├── UNSW_NB15_training-set.csv
│       └── UNSW_NB15_testing-set.csv
├── models/
│   ├── xgboost.pkl
│   ├── xgboost_label_encoders.pkl
│   └── xgboost_scaler.pkl
├── reports/
│   └── xgboost_report_*.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── data_exploration.py
│   ├── preprocess.py
│   ├── train_models.py
│   ├── evaluate.py
│   ├── hyperparameter_tuning.py
│   └── report_generator.py
├── requirements.txt
└── main.py

## 快速开始

### 环境要求

- Python 3.8+
- 依赖包见 requirements.txt

### 安装依赖

pip install -r requirements.txt

### 运行系统

python main.py

### 交互菜单

============================================================
XGBoost入侵检测系统模型训练 
============================================================
1. 数据探索 (EDA)
2. 数据预处理
3. 训练XGBoost模型
4. 评估XGBoost模型
5. 超参数调优 (优化XGBoost)
6. 生成完整模型报告
7. 运行完整流程 (1->2->3->4->6)
8. 退出

============================================================

## 数据集说明

本项目使用 UNSW-NB15 数据集，包含以下攻击类型：

- Fuzzers、Analysis、Backdoors、DoS、Exploits、Generic、Reconnaissance、Shellcode、Worms

### 主要特征

特征类别: 基础特征
示例特征: srcip, sport, dstip, dsport, proto, state

特征类别: 流量特征
示例特征: dur, sbytes, dbytes, spkts, dpkts

特征类别: 时间特征
示例特征: tcprtt, synack, ackdat, sjit, djit

特征类别: 统计特征
示例特征: ct_state_ttl, ct_srv_src, ct_dst_ltm

## 模型配置

XGBoost默认参数配置：

- 参数: n_estimators, 值: 200, 说明: 决策树数量

- 参数: max_depth, 值: 10, 说明: 树的最大深度
- 参数: learning_rate, 值: 0.05, 说明: 学习率
- 参数: subsample, 值: 0.8, 说明: 样本采样比例
- 参数: colsample_bytree, 值: 0.8, 说明: 特征采样比例
- 参数: eval_metric, 值: logloss, 说明: 评估指标

## 评估指标

报告包含以下评估指标（附通俗解释）：

- 准确率：模型判断正确的比例
- 精确率：模型判定为攻击时，实际是攻击的比例
- 召回率：所有真实攻击中被发现的比例
- F1分数：精确率和召回率的综合评分
- AUC分数：模型区分正常和攻击的能力

### 混淆矩阵分析

- 预测\实际: 正常, 攻击

- 预测正常: 正确识别, 漏报
- 预测攻击: 误报, 正确捕获

## 生成的报告

运行报告生成器后，会生成包含以下内容的Markdown报告：

1. 模型介绍和基本信息
2. 参数配置说明
3. 测试数据概况
4. 性能指标与评级
5. 混淆矩阵分析
6. 分类详细报告
7. AUC分数解读
8. 特征重要性排名
9. 错误分析与改进建议
10. 部署运维建议

## 模块说明

### config.py
- 项目路径配置
- 模型超参数
- 特征列定义

### preprocess.py
- 类别特征编码
- 缺失值处理
- 特征标准化
- 保存编码器和标准化器

### train_models.py
- XGBoost模型训练
- 进度显示和性能输出
- 模型保存

### evaluate.py
- 模型加载和预测
- 多维度指标计算
- 特征重要性提取

### hyperparameter_tuning.py
- 网格搜索参数优化
- 交叉验证评估
- 最佳模型保存

### report_generator.py
- 生成Markdown格式报告
- 通俗易懂的指标解读
- 安全等级评估
- 运维建议

## 使用建议

1. 首次使用：选择选项7运行完整流程
2. 调优模型：运行超参数调优（耗时15-30分钟）
3. 查看报告：在reports/目录查看生成的Markdown报告
4. 部署生产：使用models/xgboost.pkl进行实时检测
