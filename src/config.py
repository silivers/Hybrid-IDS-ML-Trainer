"""
配置文件 - 根据你的数据格式调整
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

# 随机森林参数
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# XGBoost参数
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

# 类别特征 - 根据你的数据格式，这些列是类别特征
CATEGORICAL_COLS = ['proto', 'service', 'state']

# 要删除的列（id列不需要用于训练）
DROP_COLS = ['id']

# 特征数量（不包括 attack_cat 和 label）
# 你的数据有 46 列，减去 id, attack_cat, label = 43 个特征
EXPECTED_FEATURES = 43