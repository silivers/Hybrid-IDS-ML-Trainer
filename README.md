# XGBoost入侵检测系统

## 项目概述
基于XGBoost算法的网络入侵检测系统模型训练，使用UNSW-NB15数据集进行训练和评估。本项目提供了完整的数据预处理、模型训练、评估和报告生成功能。

### 核心功能

- 数据探索：分析数据集结构、标签分布和攻击类型
- 数据预处理：处理类别特征、缺失值填充、特征标准化
- 模型训练：使用XGBoost算法训练入侵检测模型
- 模型评估：计算准确率、精确率、召回率、F1分数等指标
- 报告生成：自动生成Markdown格式评估报告

## 项目结构

| 文件/目录                                | 说明                        |
| ---------------------------------------- | --------------------------- |
| `XGBoost-IDS/`                           | 项目根目录                  |
| ├── `dataset/`                           | 数据集目录                  |
| │   └── `Training and Testing Sets/`     | UNSW-NB15数据集文件夹       |
| │       ├── `UNSW_NB15_training-set.csv` | 训练集数据（约17万条记录）  |
| │       └── `UNSW_NB15_testing-set.csv`  | 测试集数据（约4.5万条记录） |
| ├── `models/`                            | 训练好的模型存储目录        |
| │       ├── `xgboost.pkl`                | XGBoost模型文件             |
| │       ├── `xgboost_label_encoders.pkl` | 类别特征编码器              |
| │       ├── `xgboost_scaler.pkl`         | 特征标准化器                |
| │       └── `xgboost_feature_names.txt`  | 特征名称列表                |
| ├── `reports/`                           | 评估报告输出目录            |
| │       └── `xgboost_report_*.md`        | Markdown格式评估报告        |
| ├── `src/`                               | 源代码目录                  |
| │       ├── `config.py`                  | 配置文件                    |
| │       ├── `utils.py`                   | 工具函数                    |
| │       ├── `data_exploration.py`        | 数据探索脚本                |
| │       ├── `preprocess.py`              | 数据预处理                  |
| │       ├── `train_models.py`            | XGBoost模型训练脚本         |
| │       ├── `evaluate.py`                | 模型评估脚本                |
| │       └── `report_generator.py`        | 报告生成器                  |
| ├── `requirements.txt`                   | Python依赖包列表            |
| └── `main.py`                            | 主入口脚本                  |

## 快速开始

### 环境要求

- Python 3.8+
- 依赖包见 requirements.txt

### 安装依赖

pip install -r requirements.txt

### 运行系统

python main.py

## 数据集说明

本项目使用 UNSW-NB15 数据集，包含以下攻击类型：

- Fuzzers、Analysis、Backdoors、DoS、Exploits、Generic、Reconnaissance、Shellcode、Worms

### 主要特征

特征类别: 基础特征
示例特征: proto, service, state

特征类别: 流量特征
示例特征: sbytes, dbytes, spkts, dpkts

特征类别: 时间特征
示例特征: tcprtt, synack, ackdat, sjit, djit

特征类别: 统计特征
示例特征: ct_srv_src, ct_srv_dst, ct_dst_ltm, ct_src_ltm

## 模型配置

XGBoost默认参数配置：

| 参数 | 值 | 说明 |
|------|-----|------|
| n_estimators | 200 | 决策树数量 |
| max_depth | 10 | 树的最大深度 |
| learning_rate | 0.05 | 学习率 |
| subsample | 0.8 | 样本采样比例 |
| colsample_bytree | 0.8 | 特征采样比例 |
| eval_metric | logloss | 评估指标 |

## 评估指标

报告包含以下评估指标（附通俗解释）：

- 准确率：模型判断正确的比例
- 精确率：模型判定为攻击时，实际是攻击的比例
- 召回率：所有真实攻击中被发现的比例
- F1分数：精确率和召回率的综合评分

### 混淆矩阵分析

| 预测\实际 | 正常 | 攻击 |
|-----------|------|------|
| 预测正常 | 正确识别 | 漏报 |
| 预测攻击 | 误报 | 正确捕获 |

## 生成的报告

运行报告生成器后，会生成包含以下内容的Markdown报告：

1. 模型介绍和基本信息
2. 参数配置说明
3. 测试数据概况
4. 性能指标与评级
5. 混淆矩阵分析
6. 分类详细报告
7. 特征重要性排名
8. 简要总结与建议

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
