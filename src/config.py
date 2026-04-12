# config.py (修复版 - 更正特征名称)
"""
配置文件 - XGBoost入侵检测系统（精简版）
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 数据路径
DATA_DIR = PROJECT_ROOT / "dataset" / "Training and Testing Sets"

# 模型保存路径
MODELS_DIR = PROJECT_ROOT / "models"

# 结果输出路径（只保留 reports）
REPORTS_DIR = PROJECT_ROOT / "reports"

# 创建必要的目录
for dir_path in [MODELS_DIR, REPORTS_DIR]:
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
    'eval_metric': 'logloss',      # 评估指标
    'n_jobs': -1                   # 使用所有CPU核心
}

# 类别特征
CATEGORICAL_COLS = ['proto', 'service', 'state']

# 要删除的列
DROP_COLS = ['id']

# 特征数量（原始）
EXPECTED_FEATURES = 43

# 保留的特征列表（20个特征 - 根据实际数据修正）
SELECTED_FEATURES = [
    'proto',           # 协议类型
    'state',           # 状态
    'sbytes',          # 源到目的字节数
    'dbytes',          # 目的到源字节数
    'sttl',            # 源到目的存活时间
    'dttl',            # 目的到源存活时间
    'sloss',           # 源重传数据包数量
    'dloss',           # 目的重传数据包数量
    'spkts',           # 源到目的数据包数量 (注意: 小写)
    'dpkts',           # 目的到源数据包数量 (注意: 小写)
    'sjit',            # 源抖动 (注意: 小写)
    'djit',            # 目的抖动 (注意: 小写)
    'tcprtt',          # TCP往返时间
    'synack',          # SYN-ACK时间
    'ackdat',          # ACK-DATA时间
    'service',         # 服务类型
    'ct_srv_src',      # 相同服务到源IP的连接数
    'ct_srv_dst',      # 相同服务到目的IP的连接数
    'ct_dst_ltm',      # 目的IP连接数
    'ct_src_ltm',      # 源IP连接数
    'trans_depth',     # 管道深度
    # 'res_bdy_len',    # 响应体长度 (数据中不存在，已移除)
    'is_sm_ips_ports', # 小IP和端口
    'ct_flw_http_mthd',# HTTP方法流量
    'is_ftp_login'     # 是否FTP登录
]

# 确保类别特征在保留特征中
CATEGORICAL_COLS = [col for col in CATEGORICAL_COLS if col in SELECTED_FEATURES]

# 特征数量（精简后）
EXPECTED_FEATURES_SELECTED = len(SELECTED_FEATURES)

print(f"✅ 配置加载完成，将使用 {EXPECTED_FEATURES_SELECTED} 个特征")