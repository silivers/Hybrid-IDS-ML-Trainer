"""
配置文件 - XGBoost入侵检测系统
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 数据路径
DATA_DIR = PROJECT_ROOT / "dataset" / "Training and Testing Sets"

# 模型保存路径
MODELS_DIR = PROJECT_ROOT / "models"

# 结果输出路径
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"
LOGS_DIR = RESULTS_DIR / "logs"

# 创建必要的目录
for dir_path in [MODELS_DIR, FIGURES_DIR, REPORTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 数据文件名称
TRAIN_FILE = "UNSW_NB15_training-set.csv"
TEST_FILE = "UNSW_NB15_testing-set.csv"

# 模型参数
RANDOM_STATE = 42
CV_FOLDS = 5

# XGBoost参数（优化配置）
XGB_PARAMS = {
    'n_estimators': 200,           # 树的数量
    'max_depth': 10,               # 树的最大深度
    'learning_rate': 0.05,         # 学习率
    'subsample': 0.8,              # 样本采样比例
    'colsample_bytree': 0.8,       # 特征采样比例
    'random_state': RANDOM_STATE,  # 随机种子
    'use_label_encoder': False,    # 不使用标签编码器
    'eval_metric': 'logloss',      # 评估指标
    'n_jobs': -1                   # 使用所有CPU核心
}

# 类别特征
CATEGORICAL_COLS = ['proto', 'service', 'state']

# 要删除的列
DROP_COLS = ['id']

# 特征数量
EXPECTED_FEATURES = 43