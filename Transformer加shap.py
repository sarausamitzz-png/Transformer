# %%
import numpy as np
import pandas as pd
from torch import nn
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import warnings
import os
import shutil
import logging
import sys
import json
import math

from matplotlib import MatplotlibDeprecationWarning
from matplotlib.font_manager import FontProperties

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

"""。

说明：本段仅影响绘图外观与警告输出，不改变训练、预测与 SHAP 的数值结果。
"""

# =====================
# 1) Warning suppression（按需屏蔽输出噪声）
# =====================
warnings.filterwarnings(action="ignore", message=".*unrecognized nn.Module.*")
warnings.filterwarnings(action="ignore", message=".*Glyph.*missing from current font.*")
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=MatplotlibDeprecationWarning)
warnings.filterwarnings(action="ignore", message=".*FigureCanvasAgg is non-interactive.*")

warnings.filterwarnings(action="ignore", message="Glyph .* missing from font")

# 常见的 matplotlib/numpy 运行时告警（多出现在可视化计算/归一化/统计分布边界情况下）
warnings.filterwarnings(action="ignore", category=RuntimeWarning, message=".*divide by zero encountered.*")
warnings.filterwarnings(action="ignore", category=RuntimeWarning, message=".*invalid value encountered.*")
warnings.filterwarnings(action="ignore", category=RuntimeWarning, message=".*Mean of empty slice.*")

# 一些日志类“告警”并不走 warnings，而是 logging（例如 fontTools 子集化输出）
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)

# =====================
# 2) Font configuration（英文字体 + 中文字体 + 回退列表）
# =====================
PRIMARY_EN_FONT = "Times New Roman"
PRIMARY_CJK_FONT = "Microsoft YaHei"

plt.rcParams["font.family"] = [
    PRIMARY_EN_FONT,
    PRIMARY_CJK_FONT,
    "TeX Gyre Termes",
    "DejaVu Serif",
    "DejaVu Sans",
]

plt.rcParams["axes.unicode_minus"] = False

# =====================
# 3) Resolution & quality（DPI + 矢量输出 + 抗锯齿）
# =====================
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# 矢量格式：保留可编辑文字（PDF/SVG 更适合论文排版）
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

# 抗锯齿（线条/填充/文字）
plt.rcParams["lines.antialiased"] = True
plt.rcParams["patch.antialiased"] = True
plt.rcParams["text.antialiased"] = True

# =====================
# 4) Academic paper style（边距/刻度/字号/配色/LaTeX 数学公式）
# =====================
# 统一字号层级（可按期刊模板微调）
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10

# figure 默认边距（更适合 tight_layout / bbox_inches='tight' 的保存策略）
plt.rcParams["figure.subplot.left"] = 0.12
plt.rcParams["figure.subplot.right"] = 0.97
plt.rcParams["figure.subplot.bottom"] = 0.12
plt.rcParams["figure.subplot.top"] = 0.92
plt.rcParams["figure.subplot.wspace"] = 0.25
plt.rcParams["figure.subplot.hspace"] = 0.25
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.02

# 刻度风格：in + 四边刻度（适合论文图）
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.major.size"] = 4
plt.rcParams["ytick.major.size"] = 4
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8
plt.rcParams["xtick.minor.size"] = 2
plt.rcParams["ytick.minor.size"] = 2
plt.rcParams["xtick.minor.width"] = 0.6
plt.rcParams["ytick.minor.width"] = 0.6

# 轴线宽度
plt.rcParams["axes.linewidth"] = 0.8

# 颜色：Okabe–Ito（色盲友好，打印/灰度更稳定）；同时设置默认连续色图
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=[
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#56B4E9",
        "#E69F00",
        "#000000",
        "#999999",
    ]
)
plt.rcParams["image.cmap"] = "cividis"

# LaTeX 数学公式渲染：若系统未安装 LaTeX，则退回 mathtext（避免运行时报错）
_LATEX_OK = shutil.which("latex") is not None
plt.rcParams["text.usetex"] = bool(_LATEX_OK)
if not _LATEX_OK:
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["mathtext.rm"] = PRIMARY_EN_FONT
    plt.rcParams["mathtext.it"] = f"{PRIMARY_EN_FONT}:italic"
    plt.rcParams["mathtext.bf"] = f"{PRIMARY_EN_FONT}:bold"

device="cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体（Windows 基本都有）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*NumPy global RNG.*"
)

# %%
class MyDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EarlyStopping:
    """早停机制：如果验证集损失在patience个epoch内没有改善，则停止训练"""
    def __init__(self, patience=15, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''保存模型时打印信息'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

# %%

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def _ensure_2d_numpy(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def _build_multioutput_metrics(all_targets, all_preds):
    y_true = _ensure_2d_numpy(all_targets)
    y_pred = _ensure_2d_numpy(all_preds)

    n_targets = int(y_true.shape[1])
    mse_each = []
    rmse_each = []
    mae_each = []
    r2_each = []
    for t in range(n_targets):
        yt = y_true[:, t]
        yp = y_pred[:, t]
        mse_t = mean_squared_error(yt, yp)
        mae_t = mean_absolute_error(yt, yp)
        rmse_t = float(np.sqrt(mse_t))
        r2_t = r2_score(yt, yp)
        mse_each.append(float(mse_t))
        rmse_each.append(rmse_t)
        mae_each.append(float(mae_t))
        r2_each.append(float(r2_t))

    metrics = {
        'MSE': float(np.mean(mse_each)),
        'RMSE': float(np.mean(rmse_each)),
        'MAE': float(np.mean(mae_each)),
        'R2': float(np.mean(r2_each)),
    }

    for t in range(n_targets):
        metrics[f'MSE_t{t+1}'] = mse_each[t]
        metrics[f'RMSE_t{t+1}'] = rmse_each[t]
        metrics[f'MAE_t{t+1}'] = mae_each[t]
        metrics[f'R2_t{t+1}'] = r2_each[t]

    return metrics


def collect_predictions_matrix(model, dataloader, device, scaler_y, tta_steps=5, tta_sigma=0.01):
    """
    收集预测结果，支持 TTA (Test Time Augmentation)
    """
    model.eval()
    y_true_chunks = []
    y_pred_chunks = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            
            # TTA: 多次预测取平均
            batch_preds = []
            for _ in range(tta_steps):
                # 加入微小高斯噪声
                noise = torch.randn_like(X_batch) * tta_sigma
                X_batch_noisy = X_batch + noise
                pred = model(X_batch_noisy)
                batch_preds.append(pred.unsqueeze(0))
            
            # 计算平均预测值 (shape: batch, n_targets)
            avg_pred = torch.cat(batch_preds, dim=0).mean(dim=0)

            y_np = y_batch.detach().cpu().numpy()
            out_np = avg_pred.detach().cpu().numpy()
            y_true_chunks.append(_ensure_2d_numpy(y_np))
            y_pred_chunks.append(_ensure_2d_numpy(out_np))

    y_true = np.concatenate(y_true_chunks, axis=0) if y_true_chunks else np.zeros((0, 1), dtype=float)
    y_pred = np.concatenate(y_pred_chunks, axis=0) if y_pred_chunks else np.zeros((0, 1), dtype=float)

    if scaler_y is not None:
        y_true = scaler_y.inverse_transform(_ensure_2d_numpy(y_true))
        y_pred = scaler_y.inverse_transform(_ensure_2d_numpy(y_pred))

    return y_true, y_pred


def train(model, dataloader, criterion, optimizer, device, scaler_y=None):
    """
    训练函数
    """
    model.train()
    running_loss = 0.0
    running_mse_loss = 0.0
    all_preds = []
    all_targets = []
    
    for X_batch, y_batch in dataloader:
        # 数据转移到设备
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        y_pred = model(X_batch)

        # 多目标回归兼容：保证 y_batch/y_pred 都是二维 (batch, n_targets)
        if y_batch.dim() == 1:
            y_batch_2d = y_batch.unsqueeze(1)
        else:
            y_batch_2d = y_batch
        if y_pred.dim() == 1:
            y_pred_2d = y_pred.unsqueeze(1)
        else:
            y_pred_2d = y_pred

        loss = criterion(y_pred_2d, y_batch_2d)
        mse_loss = torch.mean((y_pred_2d - y_batch_2d) ** 2)
        
        # 反向传播
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 记录损失和预测结果
        running_loss += loss.item()
        running_mse_loss += mse_loss.item()
        all_preds.append(y_pred_2d.detach().cpu().numpy())
        all_targets.append(y_batch_2d.detach().cpu().numpy())
    
    # 计算平均损失
    avg_loss = running_loss / len(dataloader)
    avg_mse_loss = running_mse_loss / len(dataloader)

    all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, 1), dtype=float)
    all_targets = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, 1), dtype=float)
    
    # 如果提供了scaler，进行反向转换
    if scaler_y is not None:
        all_preds = scaler_y.inverse_transform(_ensure_2d_numpy(all_preds))
        all_targets = scaler_y.inverse_transform(_ensure_2d_numpy(all_targets))
    
    # 计算评估指标（既提供均值指标，也提供每个目标单独指标）
    metrics = {'loss': float(avg_loss), 'mse_loss': float(avg_mse_loss)}
    metrics.update(_build_multioutput_metrics(all_targets, all_preds))
    
    return metrics

def validate(model, dataloader, criterion, device, scaler_y=None):
    """
    验证函数
    """
    model.eval()
    running_loss = 0.0
    running_mse_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            # 数据转移到设备
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 前向传播
            y_pred = model(X_batch)

            if y_batch.dim() == 1:
                y_batch_2d = y_batch.unsqueeze(1)
            else:
                y_batch_2d = y_batch
            if y_pred.dim() == 1:
                y_pred_2d = y_pred.unsqueeze(1)
            else:
                y_pred_2d = y_pred

            loss = criterion(y_pred_2d, y_batch_2d)
            mse_loss = torch.mean((y_pred_2d - y_batch_2d) ** 2)
            
            # 记录损失和预测结果
            running_loss += loss.item()
            running_mse_loss += mse_loss.item()
            all_preds.append(y_pred_2d.detach().cpu().numpy())
            all_targets.append(y_batch_2d.detach().cpu().numpy())
    
    # 计算平均损失
    avg_loss = running_loss / len(dataloader)
    avg_mse_loss = running_mse_loss / len(dataloader)

    all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, 1), dtype=float)
    all_targets = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, 1), dtype=float)
    
    # 如果提供了scaler，进行反向转换
    if scaler_y is not None:
        all_preds = scaler_y.inverse_transform(_ensure_2d_numpy(all_preds))
        all_targets = scaler_y.inverse_transform(_ensure_2d_numpy(all_targets))
    
    # 计算评估指标（既提供均值指标，也提供每个目标单独指标）
    metrics = {'loss': float(avg_loss), 'mse_loss': float(avg_mse_loss)}
    metrics.update(_build_multioutput_metrics(all_targets, all_preds))
    
    return metrics

# %%数据输入
DATA_FILE_NAME = 'xunlian.csv'
#TARGET_COLUMN = 'MEDV'
#TARGET_INDEX = 0
csv_path = os.path.join(os.path.dirname(__file__), DATA_FILE_NAME)
#data_ = pd.read_csv(csv_path)
data_raw = pd.read_csv(csv_path, header=None)
data_ = data_raw.T 
RAW_TARGET_COL = int(os.getenv("RAW_TARGET_COL", "0"))
USE_MULTI_TARGET = bool(int(os.getenv("USE_MULTI_TARGET", "0")))
MULTI_TARGET_COLS = tuple(int(s.strip()) for s in os.getenv("MULTI_TARGET_COLS", "0,1").split(",") if str(s).strip() != "")

if USE_MULTI_TARGET:
    if len(MULTI_TARGET_COLS) < 2:
        raise ValueError(f"USE_MULTI_TARGET=1 但 MULTI_TARGET_COLS 少于2列: {MULTI_TARGET_COLS}")
    if len(set(MULTI_TARGET_COLS)) != len(MULTI_TARGET_COLS):
        raise ValueError(f"MULTI_TARGET_COLS 存在重复列: {MULTI_TARGET_COLS}")
    if min(MULTI_TARGET_COLS) < 0 or max(MULTI_TARGET_COLS) >= int(data_.shape[1]):
        raise ValueError(f"MULTI_TARGET_COLS 超出数据列范围: {MULTI_TARGET_COLS}, data_.shape={tuple(data_.shape)}")
    y = data_.iloc[:, list(MULTI_TARGET_COLS)].values
    MODEL_TARGET_INDEX = int(MULTI_TARGET_COLS.index(int(RAW_TARGET_COL))) if int(RAW_TARGET_COL) in MULTI_TARGET_COLS else 0
else:
    if int(data_.shape[1]) <= int(RAW_TARGET_COL):
        raise ValueError(f"数据列数不足，无法选择目标列 RAW_TARGET_COL={int(RAW_TARGET_COL)}: data_.shape={tuple(data_.shape)}")
    y = data_.iloc[:, int(RAW_TARGET_COL):int(RAW_TARGET_COL) + 1].values
    MODEL_TARGET_INDEX = 0
X = data_.iloc[:, 2:].values

# 多目标回归：第一列/第二列同时作为标签（两个气体浓度）
if y.ndim == 1:
    y = y.reshape(-1, 1)

PEAK1_LEN = 419
PEAK2_LEN = 450
TOTAL_EXPECTED_FEATURES = PEAK1_LEN + PEAK2_LEN + 1
if X.shape[1] != TOTAL_EXPECTED_FEATURES:
    raise ValueError(f"特征维度与预期不符: 当前 {X.shape[1]} 维, 但期望 {TOTAL_EXPECTED_FEATURES} 维 (419+450+1)")

columns = [f"Feature_{i}" for i in range(X.shape[1])]

#X = data_.drop(TARGET_COLUMN, axis=1).values
#y = data_[TARGET_COLUMN].values
#columns = data_.drop(TARGET_COLUMN, axis=1).columns.values

# %%
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# 标准化策略
# 重要：为避免数据泄漏，必须先划分训练/测试集，再只用训练集拟合 scaler（fit），最后对训练/测试分别 transform。
USE_SPLIT_SCALING = True
PRESSURE_INDEX = -1

# 这些 scaler 会在完成 train/test 划分后再进行 fit
scaler_y = None
scaler_X_spectra = None
scaler_X_pressure = None
scaler_X = None

# %%
# 从sklearn导入数据集划分工具
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
STRATIFY_N_BINS = int(os.getenv("STRATIFY_N_BINS", "10"))

_y_for_strat = np.asarray(y)[:, int(MODEL_TARGET_INDEX)].reshape(-1)
_strat_labels = None
try:
    _y_series = pd.Series(_y_for_strat)
    _unique_n = int(_y_series.nunique(dropna=True))
    _q = int(min(int(STRATIFY_N_BINS), max(2, _unique_n)))
    _bins = pd.qcut(_y_series, q=_q, labels=False, duplicates="drop")
    if int(pd.Series(_bins).nunique(dropna=True)) >= 2:
        _strat_labels = np.asarray(_bins)
except Exception:
    _strat_labels = None

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=_strat_labels,
)

y_train_raw = np.asarray(y_train).copy()
y_test_raw = np.asarray(y_test).copy()

if bool(USE_SPLIT_SCALING):
    _p_idx = int(PRESSURE_INDEX)
    if _p_idx < 0:
        p_idx = int(X_train.shape[1] + _p_idx)
    else:
        p_idx = int(_p_idx)
    p_idx = max(0, min(p_idx, int(X_train.shape[1]) - 1))
    X_train_spectra = X_train[:, :p_idx]
    X_test_spectra = X_test[:, :p_idx]
    X_train_pressure = X_train[:, p_idx:p_idx + 1]
    X_test_pressure = X_test[:, p_idx:p_idx + 1]

    scaler_X_spectra = StandardScaler() if int(X_train_spectra.shape[1]) > 0 else None
    scaler_X_pressure = StandardScaler()

    X_train_spectra_scaled = (
        scaler_X_spectra.fit_transform(X_train_spectra) if scaler_X_spectra is not None else X_train_spectra
    )
    X_test_spectra_scaled = (
        scaler_X_spectra.transform(X_test_spectra) if scaler_X_spectra is not None else X_test_spectra
    )
    X_train_pressure_scaled = scaler_X_pressure.fit_transform(X_train_pressure)
    X_test_pressure_scaled = scaler_X_pressure.transform(X_test_pressure)

    X_train = np.hstack([X_train_spectra_scaled, X_train_pressure_scaled])
    X_test = np.hstack([X_test_spectra_scaled, X_test_pressure_scaled])
else:
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(_ensure_2d_numpy(y_train))
y_test = scaler_y.transform(_ensure_2d_numpy(y_test))

N_TARGETS = int(y_train.shape[1]) if (isinstance(y_train, np.ndarray) and y_train.ndim == 2) else 1
if int(N_TARGETS) < 1:
    N_TARGETS = 1

# %%
# 设置批次大小为64
batch = 64

# 创建训练集的数据集对象和数据加载器
# 使用自定义的MyDataset类创建训练数据集
train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)

train_dataloader = data.DataLoader(train_dataset, 
                                 batch_size=batch,  # 每批处理16个样本
                                 shuffle=True)      # 随机打乱数据顺序

# 创建测试集的数据集对象和数据加载器
#test_dataset = MyDataset(test_data)    # 使用自定义的MyDataset类创建测试数据集
test_dataloader = data.DataLoader(test_dataset, 
                                batch_size=batch,   # 每批处理16个样本
                                shuffle=False)      # 测试集不需要打乱顺序

# 打印数据加载器中的批次数量
print('Train set: {} samples'.format(len(train_dataloader)))  # 显示训练集的批次数
print('Test set: {} samples'.format(len(test_dataloader)))    # 显示测试集的批次数

# 获取一个批次的训练数据进行检查
images, labels = next(iter(train_dataloader))  # 提取第一个批次的数据和标签
print('Image batch shape:', images.size())     # 打印数据批次的形状


    

# %%
import math

class PositionalEncoding(nn.Module):
    """
    位置编码模块，为输入序列添加位置信息
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PeakCNN(nn.Module):
    def __init__(self, input_len, d_model, kernel_size=35, stride=25, out_channels=128):
        super(PeakCNN, self).__init__()
        self.input_len = int(input_len)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.out_channels = int(out_channels)

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
            bias=True,
        )
        if self.out_channels == int(d_model):
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(self.out_channels, int(d_model))

    def num_tokens(self, input_len=None):
        L = int(self.input_len if input_len is None else input_len)
        if L < self.kernel_size:
            return 0
        return int((L - self.kernel_size) // self.stride + 1)

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"PeakCNN 期望输入形状为 (batch_size, length), 实际为 {x.shape}")

        x = x.unsqueeze(1)
        feat = self.conv(x)
        feat = feat.transpose(1, 2)
        tokens = self.proj(feat)
        return tokens


class FiLMLiteGammaNet(nn.Module):
    def __init__(self, cond_dim=1, hidden_dim=16, out_dim=128, gamma_max=0.5):
        super().__init__()
        self.fc1 = nn.Linear(int(cond_dim), int(hidden_dim))
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(int(hidden_dim), int(out_dim))
        self.beta_fc = nn.Linear(int(hidden_dim), int(out_dim)) # 新增：用于学习偏置 beta
        self.gamma_max = float(gamma_max)

        # 初始化为0，保证训练初期不改变原始特征
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, p):
        if p.dim() != 2:
            raise ValueError(f"FiLMLiteGammaNet 期望 p 形状为 (batch, cond_dim), 实际为 {tuple(p.shape)}")

        h = self.act(self.fc1(p))
        raw_gamma = self.fc2(h)
        gamma = torch.tanh(raw_gamma) * self.gamma_max
        beta = self.beta_fc(h) # 计算 beta
        return gamma, beta


class FiLMLitePreLNEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        film_hidden_dim=16,
        film_gamma_max=0.5,
        film_enabled=True,
    ):
        super().__init__()
        self.layer_index = None
        self.film_enabled = bool(film_enabled)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)

        self.film_att = FiLMLiteGammaNet(cond_dim=1, hidden_dim=film_hidden_dim, out_dim=d_model, gamma_max=film_gamma_max)
        self.film_mlp = FiLMLiteGammaNet(cond_dim=1, hidden_dim=film_hidden_dim, out_dim=d_model, gamma_max=film_gamma_max)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"不支持的 activation: {activation}")

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_mlp = nn.Dropout(dropout)

    def forward(self, x, p, src_mask=None, src_key_padding_mask=None, is_causal=False, **kwargs):
        return_film_stats = bool(kwargs.pop("return_film_stats", False))

        u = self.norm_attn(x)
        if self.film_enabled:
            g_att, b_att = self.film_att(p)
            g_att = g_att.unsqueeze(1)
            b_att = b_att.unsqueeze(1)
            att_scale = 1.0 + g_att
            u = att_scale * u + b_att
        else:
            att_scale = torch.ones_like(u)
            u = u
        
        att_stats = None
        if return_film_stats or getattr(self, "_film_debug_enabled", False):
            att_scale_detached = att_scale.detach()
            att_stats = {
                "mean": float(att_scale_detached.mean().cpu()),
                "min": float(att_scale_detached.min().cpu()),
                "max": float(att_scale_detached.max().cpu())
            }

        attn_out, _ = self.self_attn(
            u, u, u,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout_attn(attn_out)

        v = self.norm_mlp(x)
        if self.film_enabled:
            g_mlp, b_mlp = self.film_mlp(p)
            g_mlp = g_mlp.unsqueeze(1)
            b_mlp = b_mlp.unsqueeze(1)
            mlp_scale = 1.0 + g_mlp
            v = mlp_scale * v + b_mlp
        else:
            mlp_scale = torch.ones_like(v)
            v = v
        
        mlp_stats = None
        if return_film_stats or getattr(self, "_film_debug_enabled", False):
            mlp_scale_detached = mlp_scale.detach()
            mlp_stats = {
                "mean": float(mlp_scale_detached.mean().cpu()),
                "min": float(mlp_scale_detached.min().cpu()),
                "max": float(mlp_scale_detached.max().cpu())
            }

        ff = self.linear2(self.dropout(self.activation(self.linear1(v))))
        x = x + self.dropout_mlp(ff)
        
        if return_film_stats:
            return x, {"att": att_stats, "mlp": mlp_stats}
        return x


class FiLMLiteTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        film_hidden_dim=16,
        film_gamma_max=0.95,
        film_enabled=True,
    ):
        super().__init__()
        self.film_enabled = bool(film_enabled)
        self.film_debug_enabled = os.getenv("FILM_DEBUG", "0") == "1"
        self.film_debug_every = int(os.getenv("FILM_DEBUG_EVERY", "200"))
        if self.film_debug_every < 1:
            self.film_debug_every = 1
        self.film_debug_step = 0

        self.film_epoch_stats_enabled = os.getenv("FILM_EPOCH_STATS", "0") == "1"
        self.film_epoch_stats_csv_path = os.getenv("FILM_EPOCH_STATS_CSV", "film_epoch_stats.csv")
        self.film_epoch_stats_json_path = os.getenv("FILM_EPOCH_STATS_JSON", "film_epoch_stats.json")
        self._epoch_index = None
        self._epoch_batches = 0
        self._epoch_layer_att_sum = None
        self._epoch_layer_att_min = None
        self._epoch_layer_att_max = None
        self._epoch_layer_mlp_sum = None
        self._epoch_layer_mlp_min = None
        self._epoch_layer_mlp_max = None
        self.layers = nn.ModuleList(
            [
                FiLMLitePreLNEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    film_hidden_dim=film_hidden_dim,
                    film_gamma_max=film_gamma_max,
                    film_enabled=self.film_enabled,
                )
                for _ in range(int(num_layers))
            ]
        )
        for idx, layer in enumerate(self.layers):
            layer.layer_index = int(idx)

        self._reset_epoch_buffers()

    def _reset_epoch_buffers(self):
        n = len(self.layers)
        self._epoch_batches = 0
        self._epoch_layer_att_sum = [0.0 for _ in range(n)]
        self._epoch_layer_mlp_sum = [0.0 for _ in range(n)]
        self._epoch_layer_att_min = [math.inf for _ in range(n)]
        self._epoch_layer_mlp_min = [math.inf for _ in range(n)]
        self._epoch_layer_att_max = [-math.inf for _ in range(n)]
        self._epoch_layer_mlp_max = [-math.inf for _ in range(n)]

    def begin_epoch(self, epoch_index: int):
        self._epoch_index = int(epoch_index)
        self._reset_epoch_buffers()

    def end_epoch(self):
        if not self.film_epoch_stats_enabled:
            return None
        if self._epoch_index is None:
            return None
        if int(self._epoch_batches) <= 0:
            return None

        n = len(self.layers)
        att_layer_means = [self._epoch_layer_att_sum[i] / float(self._epoch_batches) for i in range(n)]
        mlp_layer_means = [self._epoch_layer_mlp_sum[i] / float(self._epoch_batches) for i in range(n)]

        overall_att_mean = float(sum(att_layer_means) / float(n))
        overall_mlp_mean = float(sum(mlp_layer_means) / float(n))
        overall_att_min = float(min(self._epoch_layer_att_min))
        overall_att_max = float(max(self._epoch_layer_att_max))
        overall_mlp_min = float(min(self._epoch_layer_mlp_min))
        overall_mlp_max = float(max(self._epoch_layer_mlp_max))

        record = {
            "epoch": int(self._epoch_index),
            "batches": int(self._epoch_batches),
            "att_scale_mean": overall_att_mean,
            "att_scale_min": overall_att_min,
            "att_scale_max": overall_att_max,
            "mlp_scale_mean": overall_mlp_mean,
            "mlp_scale_min": overall_mlp_min,
            "mlp_scale_max": overall_mlp_max,
            "per_layer": [
                {
                    "layer": int(i),
                    "att_scale_mean": float(att_layer_means[i]),
                    "att_scale_min": float(self._epoch_layer_att_min[i]),
                    "att_scale_max": float(self._epoch_layer_att_max[i]),
                    "mlp_scale_mean": float(mlp_layer_means[i]),
                    "mlp_scale_min": float(self._epoch_layer_mlp_min[i]),
                    "mlp_scale_max": float(self._epoch_layer_mlp_max[i]),
                }
                for i in range(n)
            ],
        }

        try:
            row = {
                "epoch": record["epoch"],
                "batches": record["batches"],
                "att_scale_mean": record["att_scale_mean"],
                "att_scale_min": record["att_scale_min"],
                "att_scale_max": record["att_scale_max"],
                "mlp_scale_mean": record["mlp_scale_mean"],
                "mlp_scale_min": record["mlp_scale_min"],
                "mlp_scale_max": record["mlp_scale_max"],
            }
            df_row = pd.DataFrame([row])
            write_header = not os.path.exists(self.film_epoch_stats_csv_path)
            df_row.to_csv(self.film_epoch_stats_csv_path, mode="a", header=write_header, index=False, encoding="utf-8-sig")
        except Exception:
            pass

        try:
            history = []
            if os.path.exists(self.film_epoch_stats_json_path):
                with open(self.film_epoch_stats_json_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    if not isinstance(history, list):
                        history = []
            history.append(record)
            with open(self.film_epoch_stats_json_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return record

    def forward(self, src, p, src_mask=None, src_key_padding_mask=None, is_causal=False, **kwargs):
        out = src
        if self.training:
            self.film_debug_step += 1
        debug_on = bool(self.film_debug_enabled) and bool(self.training)

        collect_epoch_stats = bool(self.film_epoch_stats_enabled) and bool(self.training)
        if collect_epoch_stats:
            if self._epoch_index is None:
                self.begin_epoch(0)
            self._epoch_batches += 1

        for layer in self.layers:
            layer._film_debug_enabled = debug_on
            layer._film_debug_step = int(self.film_debug_step)
            layer._film_debug_every = int(self.film_debug_every)
            if collect_epoch_stats:
                out, film_stats = layer(
                    out,
                    p,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    is_causal=is_causal,
                    return_film_stats=True,
                )
                li = int(layer.layer_index) if layer.layer_index is not None else None
                if li is not None and film_stats is not None:
                    att = film_stats.get("att")
                    mlp = film_stats.get("mlp")
                    if att is not None:
                        self._epoch_layer_att_sum[li] += float(att["mean"])
                        self._epoch_layer_att_min[li] = min(self._epoch_layer_att_min[li], float(att["min"]))
                        self._epoch_layer_att_max[li] = max(self._epoch_layer_att_max[li], float(att["max"]))
                    if mlp is not None:
                        self._epoch_layer_mlp_sum[li] += float(mlp["mean"])
                        self._epoch_layer_mlp_min[li] = min(self._epoch_layer_mlp_min[li], float(mlp["min"]))
                        self._epoch_layer_mlp_max[li] = max(self._epoch_layer_mlp_max[li], float(mlp["max"]))
            else:
                out = layer(
                    out,
                    p,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    is_causal=is_causal,
                )
        return out


class SpectraInteractionRegressor(nn.Module):
    def __init__(
        self,
        peak1_len,
        peak2_len,
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        output_dim=1,
        cnn_kernel_size=35,
        cnn_stride=25,
        cnn_out_channels=128,
        film_hidden_dim=16,
        film_gamma_max=0.95,
        film_enabled=True,
        use_positional_encoding=True,
    ):
        super(SpectraInteractionRegressor, self).__init__()
        self.peak1_len = peak1_len
        self.peak2_len = peak2_len
        self.d_model = d_model

        self.peak1_cnn = PeakCNN(
            peak1_len,
            d_model,
            kernel_size=cnn_kernel_size,
            stride=cnn_stride,
            out_channels=cnn_out_channels,
        )
        self.peak2_cnn = PeakCNN(
            peak2_len,
            d_model,
            kernel_size=cnn_kernel_size,
            stride=cnn_stride,
            out_channels=cnn_out_channels,
        )
        self.token_type_embeddings = nn.Embedding(2, d_model)

        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=5000)
        else:
            self.pos_encoder = None

        self.transformer_encoder = FiLMLiteTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            film_hidden_dim=film_hidden_dim,
            film_gamma_max=film_gamma_max,
            film_enabled=film_enabled,
        )

        # 多目标回归：共享编码器/特征抽取层，仅在最后使用多个独立输出头
        hidden_dim = dim_feedforward // 2
        self.shared_output_trunk = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(int(output_dim))])

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"期望输入形状为 (batch_size, feature_dim), 实际为 {x.shape}")

        if x.shape[1] < int(self.peak1_len) + 1:
            raise ValueError(
                f"输入特征维度不足: feature_dim={x.shape[1]}, 需要至少 peak1_len+1={int(self.peak1_len)+1}"
            )

        peak1 = x[:, : self.peak1_len]
        peak2 = x[:, self.peak1_len : -1]
        pressure = x[:, -1:].contiguous()

        tok1 = self.peak1_cnn(peak1)
        tok2 = self.peak2_cnn(peak2)
        tokens = torch.cat([tok1, tok2], dim=1)

        n1 = int(tok1.shape[1])
        n2 = int(tok2.shape[1])
        batch_size = x.size(0)
        type_ids = torch.cat(
            [
                torch.zeros(n1, dtype=torch.long, device=x.device),
                torch.ones(n2, dtype=torch.long, device=x.device),
            ],
            dim=0,
        ).unsqueeze(0).expand(batch_size, -1)
        tokens = tokens + self.token_type_embeddings(type_ids)

        if self.pos_encoder is not None:
            tokens = tokens.transpose(0, 1)
            tokens = self.pos_encoder(tokens)
            tokens = tokens.transpose(0, 1)

        encoded = self.transformer_encoder(tokens, pressure)
        pooled = encoded.mean(dim=1)
        feat = self.shared_output_trunk(pooled)
        outs = [head(feat) for head in self.output_heads]
        if len(outs) == 1:
            return outs[0]
        return torch.cat(outs, dim=1)

# %%
# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FILM_ENABLED = bool(int(os.getenv("FILM_ENABLED", "1")))

# 实例化包含峰1/峰2/压力三支路 + Transformer 的回归模型
model = SpectraInteractionRegressor(
    peak1_len=PEAK1_LEN,
    peak2_len=PEAK2_LEN,
    d_model=128,
    nhead=8,
    num_layers=6,           
    dim_feedforward=2048,   
    dropout=0.15,            # 增加 Dropout (0.1 -> 0.2) 以增强泛化能力，防止过拟合
    output_dim=N_TARGETS,
    film_hidden_dim=512,    
    film_gamma_max=1.5,
    film_enabled=FILM_ENABLED
)

from torchinfo import summary  # 导入torchinfo库的summary函数，用于查看模型结构

# 定义输入数据的形状
input_size = (1, len(columns))  # 表示：
                         # 1: 批次大小（batch size）
                         # 13: 特征维度（feature dimension）

# 使用torchinfo的summary函数查看模型详情
summary(model, input_size=input_size)

model = model.to(device)

# %%
import copy
import time
import torch
import torch.nn as nn
import numpy as np

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设置优化器为AdamW，学习率为2e-4，增加权重衰减以防止过拟合
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, betas=(0.8500000000, 0.9000000000), weight_decay=0.0001, eps=0.0000000081)

# 学习率调度器：使用余弦退火重启策略 (CosineAnnealingWarmRestarts)
# 注意：该调度器应按“epoch/step”推进，而非按验证损失推进
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

# 早停机制初始化 (patience 50 -> 100，配合重启策略适当放宽)
early_stopping = EarlyStopping(patience=60, verbose=True)
model_save_path = './best_transformer_model.pth'

# 使用Huber Loss，对异常值更鲁棒，减少极端偏差的影响
loss_fn = nn.HuberLoss(delta=1.0)

epochs = 300  # 保持训练轮数
R2_STOP_THRESHOLD = 0.9999

# 初始化存储训练过程中的指标列表
train_loss = []  # 存储训练损失
train_r2 = []    # 存储训练R²
val_loss = []    # 存储验证损失
val_r2 = []      # 存储验证R²
train_mse_loss = []  # 存储训练阶段MSE损失
val_mse_loss = []    # 存储验证阶段MSE损失
lr_history = []      # 存储每个epoch的学习率

export_dir_epoch = os.getenv("EXPORT_DIR", os.path.join(os.path.dirname(__file__), "shuju"))
os.makedirs(export_dir_epoch, exist_ok=True)
epoch_metrics_csv_path = os.path.join(export_dir_epoch, "epoch_train_val_metrics.csv")


def save_training_strategy_plots(export_dir, train_huber_hist, val_huber_hist, train_mse_hist, val_mse_hist, lr_hist):
    """
    保存训练优化策略相关图：
    1. 学习率随epoch变化曲线
    2. Huber Loss 与 MSE Loss 的迭代对比曲线
    """
    if len(train_huber_hist) == 0:
        return

    epochs_axis = np.arange(1, len(train_huber_hist) + 1, dtype=int)

    from matplotlib.ticker import ScalarFormatter

    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(epochs_axis, lr_hist, color="#1565C0", linewidth=2.2, label="Learning Rate")
    plt.scatter(epochs_axis, lr_hist, s=12, color="#EF6C00", zorder=3)
    plt.xlabel("Epoch", fontsize=14, fontweight="bold")
    plt.ylabel("Learning Rate", fontsize=14, fontweight="bold")
    plt.title("CosineAnnealingWarmRestarts Learning Rate Schedule", fontsize=15, fontweight="bold")
    plt.tick_params(axis="both", labelsize=12, width=1.5, length=5, direction="in")
    plt.grid(True, linestyle="--", alpha=0.25)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")
    # 关闭offset显示，避免出现“1e-10+1.9999e-4”这类不直观标注
    ax = plt.gca()
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    fmt = ScalarFormatter(useMathText=False)
    fmt.set_scientific(False)
    fmt.set_useOffset(False)
    ax.yaxis.set_major_formatter(fmt)
    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, "lr_schedule_vs_epoch.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(export_dir, "lr_schedule_vs_epoch.pdf"), dpi=600, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    axes[0].plot(epochs_axis, train_huber_hist, color="#1E88E5", linewidth=2.0, label="Train Huber")
    axes[0].plot(epochs_axis, val_huber_hist, color="#E53935", linewidth=2.0, label="Val Huber")
    axes[0].set_title("Huber Loss Curve", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Huber Loss", fontsize=13, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.25)

    axes[1].plot(epochs_axis, train_mse_hist, color="#43A047", linewidth=2.0, label="Train MSE")
    axes[1].plot(epochs_axis, val_mse_hist, color="#FB8C00", linewidth=2.0, label="Val MSE")
    axes[1].set_title("MSE Loss Curve", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("MSE Loss", fontsize=13, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.25)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=11, width=1.5, length=5, direction="in")
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color("black")

    fig.suptitle("Huber vs MSE Loss Comparison", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(export_dir, "huber_vs_mse_loss_curves.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(export_dir, "huber_vs_mse_loss_curves.pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig)

# 记录训练开始时间
start_time = time.time()

# 开始训练循环
for epoch in range(epochs):
    if hasattr(model, "transformer_encoder") and hasattr(model.transformer_encoder, "begin_epoch"):
        model.transformer_encoder.begin_epoch(epoch + 1)
    model.train()  # 设置模型为训练模式
    train_metrics = train(model, train_dataloader, loss_fn, optimizer, device, scaler_y)  # 训练一个epoch
    
    model.eval()   # 设置模型为评估模式
    val_metrics = validate(model, test_dataloader, loss_fn, device, scaler_y)  # 验证模型性能

    # 获取R²和损失值（R2 为多目标均值；同时保留每个目标单独R2）
    epoch_train_r2 = train_metrics['R2']
    epoch_train_loss = train_metrics['loss']
    epoch_train_mse_loss = train_metrics.get('mse_loss', np.nan)
    epoch_val_r2 = val_metrics['R2']
    epoch_val_loss = val_metrics['loss']
    epoch_val_mse_loss = val_metrics.get('mse_loss', np.nan)

    # 更新学习率调度器（按epoch推进，符合余弦退火重启语义）
    scheduler.step(epoch + 1)

    # 记录训练过程中的各项指标
    train_r2.append(epoch_train_r2)
    train_loss.append(epoch_train_loss)
    val_r2.append(epoch_val_r2)
    val_loss.append(epoch_val_loss)
    train_mse_loss.append(epoch_train_mse_loss)
    val_mse_loss.append(epoch_val_mse_loss)
    lr_history.append(float(optimizer.state_dict()['param_groups'][0]['lr']))

    pd.DataFrame({
        "epoch": np.arange(1, len(train_loss) + 1, dtype=int),
        "train_loss": np.asarray(train_loss, dtype=float),
        "train_r2": np.asarray(train_r2, dtype=float),
        "val_loss": np.asarray(val_loss, dtype=float),
        "val_r2": np.asarray(val_r2, dtype=float),
        "train_mse_loss": np.asarray(train_mse_loss, dtype=float),
        "val_mse_loss": np.asarray(val_mse_loss, dtype=float),
        "lr": np.asarray(lr_history, dtype=float),
    }).to_csv(epoch_metrics_csv_path, index=False, encoding="utf-8-sig")

    # 获取当前学习率并打印训练信息
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    if N_TARGETS >= 2:
        template = (
            'Epoch:{:2d}, '
            'Train_R2_mean:{:.4f}, Train_R2_t1:{:.4f}, Train_R2_t2:{:.4f}, Train_loss:{:.6f}, '
            'Val_R2_mean:{:.4f}, Val_R2_t1:{:.4f}, Val_R2_t2:{:.4f}, Val_loss:{:.6f}, Lr:{:.6f}'
        )
        print(template.format(
            epoch + 1,
            epoch_train_r2,
            float(train_metrics.get('R2_t1', epoch_train_r2)),
            float(train_metrics.get('R2_t2', epoch_train_r2)),
            epoch_train_loss,
            epoch_val_r2,
            float(val_metrics.get('R2_t1', epoch_val_r2)),
            float(val_metrics.get('R2_t2', epoch_val_r2)),
            epoch_val_loss,
            lr,
        ))
    else:
        template = ('Epoch:{:2d}, Train_R2:{:.4f}, Train_loss:{:.6f}, Val_R2:{:.4f}, Val_loss:{:.6f}, Lr:{:.6f}')
        print(template.format(epoch+1, epoch_train_r2, epoch_train_loss, epoch_val_r2, epoch_val_loss, lr))

    if hasattr(model, "transformer_encoder") and hasattr(model.transformer_encoder, "end_epoch"):
        model.transformer_encoder.end_epoch()

    if float(epoch_val_r2) >= float(R2_STOP_THRESHOLD):
        print(f"Val_R2 达到阈值 {float(R2_STOP_THRESHOLD):.2f}，提前停止训练并保存模型...")
        torch.save(model.state_dict(), model_save_path)
        break

    # 早停检查
    early_stopping(epoch_val_loss, model, model_save_path)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

# 训练结束，加载最佳模型
print("Loading best model from early stopping checkpoint...")
model.load_state_dict(torch.load(model_save_path, map_location=device))
best_model = model # 更新best_model为加载的最佳模型

# 训练结束，计算总训练时间
end_time = time.time()
execution_time = end_time - start_time
print("执行时间：", execution_time, "秒")


save_training_strategy_plots(
    export_dir=export_dir_epoch,
    train_huber_hist=train_loss,
    val_huber_hist=val_loss,
    train_mse_hist=train_mse_loss,
    val_mse_hist=val_mse_loss,
    lr_hist=lr_history,
)

# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import explained_variance_score, max_error
import numpy as np

def evaluate_model_performance(model, train_loader, test_loader, device, scaler_y):
    """
    评估模型在训练集和测试集上的多个性能指标
    """
    train_true, train_pred = collect_predictions_matrix(model, train_loader, device, scaler_y)
    test_true, test_pred = collect_predictions_matrix(model, test_loader, device, scaler_y)

    n_targets = int(train_true.shape[1]) if train_true.ndim == 2 else 1

    def _metrics_for_pair(y_true_arr, y_pred_arr):
        y_true_arr = _ensure_2d_numpy(y_true_arr)
        y_pred_arr = _ensure_2d_numpy(y_pred_arr)
        per_target = []
        for t in range(y_true_arr.shape[1]):
            yt = y_true_arr[:, t]
            yp = y_pred_arr[:, t]
            mse = mean_squared_error(yt, yp)
            rmse = float(np.sqrt(mse))
            mae = mean_absolute_error(yt, yp)
            r2 = r2_score(yt, yp)
            evs = explained_variance_score(yt, yp)
            maxerr = max_error(yt, yp)
            denom = np.where(np.abs(yt) < 1e-12, 1e-12, yt)
            mape = float(np.mean(np.abs((yt - yp) / denom)) * 100)
            per_target.append({
                'MSE': float(mse),
                'RMSE': rmse,
                'MAE': float(mae),
                'R2': float(r2),
                'EVS': float(evs),
                'MAX_ERROR': float(maxerr),
                'MAPE': mape,
            })
        mean_metrics = {k: float(np.mean([d[k] for d in per_target])) for k in per_target[0].keys()} if per_target else {}
        return mean_metrics, per_target

    train_mean, train_each = _metrics_for_pair(train_true, train_pred)
    test_mean, test_each = _metrics_for_pair(test_true, test_pred)

    metrics = {
        'MSE (均方误差)': [train_mean.get('MSE', 0.0), test_mean.get('MSE', 0.0)],
        'RMSE (均方根误差)': [train_mean.get('RMSE', 0.0), test_mean.get('RMSE', 0.0)],
        'MAE (平均绝对误差)': [train_mean.get('MAE', 0.0), test_mean.get('MAE', 0.0)],
        'R² (决定系数)': [train_mean.get('R2', 0.0), test_mean.get('R2', 0.0)],
        'EVS (解释方差分)': [train_mean.get('EVS', 0.0), test_mean.get('EVS', 0.0)],
        'MAX_ERROR (最大误差)': [train_mean.get('MAX_ERROR', 0.0), test_mean.get('MAX_ERROR', 0.0)],
        'MAPE (平均绝对百分比误差)': [train_mean.get('MAPE', 0.0), test_mean.get('MAPE', 0.0)]
    }

    export_dir = os.getenv("EXPORT_DIR", os.path.join(os.path.dirname(__file__), "shuju"))
    os.makedirs(export_dir, exist_ok=True)

    def _pred_frame(y_true_arr, y_pred_arr):
        y_true_arr = _ensure_2d_numpy(y_true_arr)
        y_pred_arr = _ensure_2d_numpy(y_pred_arr)
        out = {"row_id": np.arange(y_true_arr.shape[0], dtype=int)}
        for t in range(int(y_true_arr.shape[1])):
            out[f"y_true_t{t}"] = y_true_arr[:, t]
            out[f"y_pred_t{t}"] = y_pred_arr[:, t]
            out[f"abs_err_t{t}"] = np.abs(y_true_arr[:, t] - y_pred_arr[:, t])
        return pd.DataFrame(out)

    df_train_pred = _pred_frame(train_true, train_pred)
    df_test_pred = _pred_frame(test_true, test_pred)
    df_train_pred.to_csv(os.path.join(export_dir, "train_true_pred.csv"), index=False, encoding="utf-8-sig")
    df_test_pred.to_csv(os.path.join(export_dir, "test_true_pred.csv"), index=False, encoding="utf-8-sig")

    metrics_rows = []
    for metric_name, (train_value, test_value) in metrics.items():
        metrics_rows.append({"metric": str(metric_name), "train": float(train_value), "test": float(test_value)})
    pd.DataFrame(metrics_rows).to_csv(os.path.join(export_dir, "metrics_summary.csv"), index=False, encoding="utf-8-sig")

    each_rows = []
    for t in range(int(n_targets)):
        each_rows.append({"split": "train", "target": int(t), **train_each[t]})
        each_rows.append({"split": "test", "target": int(t), **test_each[t]})
    pd.DataFrame(each_rows).to_csv(os.path.join(export_dir, "metrics_each_target.csv"), index=False, encoding="utf-8-sig")
    
    # 打印结果表格
    print("\n模型性能评估:")
    print("-" * 80)
    print(f"{'指标':<25} {'训练集':<25} {'测试集':<25}")
    print("-" * 80)
    
    for metric, (train_value, test_value) in metrics.items():
        print(f"{metric:<25} {train_value:>25.4f} {test_value:>25.4f}")
    
    print("-" * 80)
    
    # 绘制性能指标对比图
    plt.figure(figsize=(12, 6))
    
    # 准备数据
    metrics_names = list(metrics.keys())
    train_values = [metrics[m][0] for m in metrics_names]
    test_values = [metrics[m][1] for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    # 绘制柱状图
    plt.bar(x - width/2, train_values, width, label='训练集', color='#3498db', alpha=0.8)
    plt.bar(x + width/2, test_values, width, label='测试集', color='#e74c3c', alpha=0.8)
    
    plt.xlabel('评估指标')
    plt.ylabel('值')
    plt.title('模型性能指标对比')
    plt.xticks(x, metrics_names, rotation=45, ha='right')
    plt.legend()
    
    # 在柱状图上添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', rotation=0)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    if n_targets >= 2:
        print("\n分目标性能指标（每个气体单独计算）:")
        for t in range(n_targets):
            print(f"- 目标{t+1} 训练集: R2={train_each[t]['R2']:.6f}, RMSE={train_each[t]['RMSE']:.6f}, MAE={train_each[t]['MAE']:.6f}, MAPE={train_each[t]['MAPE']:.6f}%")
            print(f"        测试集: R2={test_each[t]['R2']:.6f}, RMSE={test_each[t]['RMSE']:.6f}, MAE={test_each[t]['MAE']:.6f}, MAPE={test_each[t]['MAPE']:.6f}%")

# 使用函数评估模型性能
evaluate_model_performance(best_model, train_dataloader, test_dataloader, device, scaler_y)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 获取训练集和测试集的预测值和真实值
def get_predictions(model, dataloader, device, scaler_y):
    y_true, y_pred = collect_predictions_matrix(model, dataloader, device, scaler_y)
    if y_true.ndim == 2 and int(y_true.shape[1]) == 1:
        return y_true[:, 0], y_pred[:, 0]
    return y_true, y_pred

# 密度热力图回归散点图
def plot_regression_results(model, train_loader, test_loader, device, scaler_y):
    train_true, train_pred = get_predictions(model, train_loader, device, scaler_y)
    test_true, test_pred = get_predictions(model, test_loader, device, scaler_y)

    if isinstance(train_true, np.ndarray) and train_true.ndim == 2:
        n_targets = int(train_true.shape[1])
        ncols = 2 if n_targets > 1 else 1
        nrows = int(np.ceil(n_targets / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows))
        axes = np.atleast_1d(axes).reshape(-1)

        for t in range(n_targets):
            ax = axes[t]
            all_true = np.concatenate([train_true[:, t], test_true[:, t]])
            all_pred = np.concatenate([train_pred[:, t], test_pred[:, t]])

            mae = mean_absolute_error(all_true, all_pred)
            mse = mean_squared_error(all_true, all_pred)
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(all_true, all_pred)) if all_true.size >= 2 else 0.0

            xy = np.vstack([all_true, all_pred])
            z = gaussian_kde(xy)(xy) if all_true.size >= 2 else np.zeros_like(all_true, dtype=float)
            idx = np.argsort(z)
            x_sorted, y_sorted, z_sorted = all_true[idx], all_pred[idx], z[idx]

            sc = ax.scatter(x_sorted, y_sorted, c=z_sorted, s=45, alpha=0.8, cmap='inferno', edgecolors='none')
            cbar = fig.colorbar(sc, ax=ax, shrink=1, aspect=20)
            cbar.set_label('Probability Density', fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)

            min_val = float(min(np.min(all_true), np.min(all_pred))) if all_true.size else 0.0
            max_val = float(max(np.max(all_true), np.max(all_pred))) if all_true.size else 1.0
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8, label='1:1 line')

            ax.set_xlabel('True Values', fontsize=14, fontweight='bold')
            ax.set_ylabel('Predict Values', fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', labelsize=12, width=2, length=6)

            stats_text = f'Target {t+1}\nMAE = {mae:.5f}\nMSE = {mse:.5f}\nRMSE = {rmse:.5f}\nR$^2$ = {r2:.5f}'
            ax.text(
                0.05, 0.80, stats_text, transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                bbox=dict(facecolor='none', edgecolor='none') # 移除统计信息背景
            )
            ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(0.98, 0.92), frameon=False)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_color('black')
            ax.grid(False) # 移除网格线
            ax.set_aspect('equal', adjustable='box')

        for k in range(n_targets, len(axes)):
            axes[k].axis('off')

        plt.tight_layout()
        plt.show()
        return

    all_true = np.concatenate([train_true, test_true])
    all_pred = np.concatenate([train_pred, test_pred])
    mae = mean_absolute_error(all_true, all_pred)
    mse = mean_squared_error(all_true, all_pred)
    rmse = np.sqrt(mse)
    r2 = float(r2_score(all_true, all_pred)) if all_true.size >= 2 else 0.0

    plt.figure(figsize=(10, 8))

    xy = np.vstack([all_true, all_pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x_sorted, y_sorted, z_sorted = all_true[idx], all_pred[idx], z[idx]

    scatter = plt.scatter(x_sorted, y_sorted, c=z_sorted, s=50, alpha=0.8, cmap='inferno', edgecolors='none')

    cbar = plt.colorbar(scatter, shrink=1, aspect=20)
    cbar.set_label('Probability Density', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    min_val = min(np.min(all_true), np.min(all_pred))
    max_val = max(np.max(all_true), np.max(all_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8, label='1:1 line')

    plt.xlabel('True Values', fontsize=16, fontweight='bold')
    plt.ylabel('Predict Values', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')

    stats_text = f'MAE = {mae:.5f}\nMSE = {mse:.5f}\nRMSE = {rmse:.5f}\nR$^2$ = {r2:.5f}'
    plt.text(
        0.05, 0.85, stats_text, transform=plt.gca().transAxes,
        fontsize=14, fontweight='bold',
        bbox=dict(facecolor='none', edgecolor='none') # 移除统计信息背景
    )

    plt.legend(fontsize=12, loc='upper right', bbox_to_anchor=(0.98, 0.92), frameon=False)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    ax.tick_params(width=2, length=6)
    plt.grid(False) # 移除网格线
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# 绘制阶梯图：真实值（Set Value）与预测值（Decoupling model）对比
def plot_step_chart(model, test_loader, device, scaler_y, export_dir):
    y_true, y_pred = get_predictions(model, test_loader, device, scaler_y)
    
    # 确保是一维数组
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
        
    # 按真实值排序
    sorted_indices = np.argsort(y_true)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    sample_points = np.arange(len(y_true_sorted))
    
    plt.figure(figsize=(10, 8), dpi=600)
    
    # 绘制真实值（黑色连线+圆点）
    plt.plot(sample_points, y_true_sorted, color='black', marker='o', markersize=6, 
             linestyle='-', linewidth=2, label='Set Value', alpha=0.8)
             
    # 绘制预测值（蓝色连线+圆点，略带透明度）
    plt.plot(sample_points, y_pred_sorted, color='#0099ff', marker='o', markersize=6,
             linestyle='-', linewidth=2, label='Decoupling model', alpha=0.8)
    
    # 样式设置
    plt.xlabel('Sample Point (n)', fontsize=16, fontweight='bold')
    plt.ylabel('Concentration (ppm)', fontsize=16, fontweight='bold')
    
    # 刻度设置
    plt.tick_params(axis='both', which='major', labelsize=14, direction='in', width=2, length=6)
    
    # 图例设置
    plt.legend(loc='upper left', fontsize=14, frameon=False, prop={'weight':'bold', 'size':14})
    
    # 边框加粗
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
        
    # 添加 (a) 标签
    plt.text(0.02, 0.92, '(a)', transform=ax.transAxes, fontsize=16, fontweight='bold')
    
    # 导出阶梯图数据到 CSV
    df_step_chart = pd.DataFrame({
        'Sample Point': sample_points,
        'Set Value': y_true_sorted,
        'Decoupling model': y_pred_sorted
    })
    csv_save_path = os.path.join(export_dir, 'step_chart_data.csv')
    df_step_chart.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 阶梯对比图原始数据已导出至: {csv_save_path}")

    # 保存图片
    save_path = os.path.join(export_dir, 'step_chart_comparison.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"✅ 阶梯对比图已保存至: {save_path}")
    plt.show()

def plot_mre_by_concentration_interval(model, test_loader, device, scaler_y, export_dir):
    y_true, y_pred = get_predictions(model, test_loader, device, scaler_y)

    if isinstance(y_true, np.ndarray) and y_true.ndim > 1:
        y_true = y_true[:, 0]
    if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
        y_pred = y_pred[:, 0]

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = np.where(np.abs(y_true) < 1e-12, 1e-12, np.abs(y_true))
    re_pct = np.abs(y_pred - y_true) / denom * 100.0

    rounded_true = np.round(y_true, 6)
    unique_levels = np.unique(rounded_true)
    if unique_levels.size <= 20:
        level_order = np.sort(unique_levels)
        rows = []
        point_rows = []
        for i, lv in enumerate(level_order):
            m = (rounded_true == lv)
            re_lv = re_pct[m]
            rows.append({
                "interval_index": int(i),
                "set_value_ppm": float(lv),
                "n_samples": int(np.sum(m)),
                "mre_percent": float(np.mean(re_lv)),
                "re_std_percent": float(np.std(re_lv)),
            })
            for v in re_lv:
                point_rows.append({
                    "interval_index": int(i),
                    "set_value_ppm": float(lv),
                    "re_percent": float(v),
                })
        df_summary = pd.DataFrame(rows)
        df_points = pd.DataFrame(point_rows)
        x_labels = [f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.2f}" for v in df_summary["set_value_ppm"].tolist()]
    else:
        n_bins = 8
        bins = np.linspace(float(np.min(y_true)), float(np.max(y_true)), n_bins + 1)
        bin_idx = np.digitize(y_true, bins[1:-1], right=False)
        rows = []
        point_rows = []
        for i in range(n_bins):
            m = (bin_idx == i)
            if not np.any(m):
                continue
            re_lv = re_pct[m]
            lo = float(bins[i])
            hi = float(bins[i + 1])
            rows.append({
                "interval_index": int(i),
                "interval_low_ppm": lo,
                "interval_high_ppm": hi,
                "n_samples": int(np.sum(m)),
                "mre_percent": float(np.mean(re_lv)),
                "re_std_percent": float(np.std(re_lv)),
            })
            for v in re_lv:
                point_rows.append({
                    "interval_index": int(i),
                    "interval_low_ppm": lo,
                    "interval_high_ppm": hi,
                    "re_percent": float(v),
                })
        df_summary = pd.DataFrame(rows)
        df_points = pd.DataFrame(point_rows)
        x_labels = [f"{r['interval_low_ppm']:.0f}-{r['interval_high_ppm']:.0f}" for _, r in df_summary.iterrows()]

    x = np.arange(len(df_summary))
    y_bar = df_summary["mre_percent"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=600)
    bars = ax.bar(x, y_bar, color="#4C72B0", alpha=0.88, width=0.65, label="MRE by interval")

    rng = np.random.default_rng(42)
    for i in range(len(df_summary)):
        yp = df_points.loc[df_points["interval_index"] == int(df_summary.iloc[i]["interval_index"]), "re_percent"].to_numpy(dtype=float)
        if yp.size == 0:
            continue
        jitter = rng.uniform(-0.2, 0.2, size=yp.size)
        ax.scatter(np.full(yp.size, x[i]) + jitter, yp, s=20, color="#E24A33", alpha=0.65, edgecolors="none", label="Sample RE" if i == 0 else None)

    for rect, v in zip(bars, y_bar):
        ax.text(rect.get_x() + rect.get_width() / 2.0, rect.get_height(), f"{v:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Concentration Interval (ppm)", fontsize=15, fontweight="bold")
    ax.set_ylabel("MRE (%)", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12, direction="in", width=1.8, length=5)
    ax.tick_params(axis="x", direction="in", width=1.8, length=5)
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color("black")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()

    csv_summary = os.path.join(export_dir, "mre_by_concentration_interval.csv")
    csv_points = os.path.join(export_dir, "mre_by_concentration_interval_points.csv")
    fig_path = os.path.join(export_dir, "mre_by_concentration_interval.png")
    # 每次运行额外在脚本同级目录保存一份PDF，便于答辩/论文直接使用
    fixed_pdf_path = os.path.join(os.path.dirname(__file__), "mre_by_concentration_interval.pdf")
    df_summary.to_csv(csv_summary, index=False, encoding="utf-8-sig")
    df_points.to_csv(csv_points, index=False, encoding="utf-8-sig")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.savefig(fixed_pdf_path, format="pdf", dpi=1200, bbox_inches="tight")
    print(f"✅ 各浓度区间MRE汇总已导出至: {csv_summary}")
    print(f"✅ 各样本RE散点数据已导出至: {csv_points}")
    print(f"✅ 各浓度区间MRE柱散图已保存至: {fig_path}")
    print(f"✅ 各浓度区间MRE柱散图PDF已保存至: {fixed_pdf_path}")
    plt.show()

# 使用函数绘制图形
plot_regression_results(best_model, train_dataloader, test_dataloader, device, scaler_y)
plot_step_chart(best_model, test_dataloader, device, scaler_y, export_dir_epoch)
plot_mre_by_concentration_interval(best_model, test_dataloader, device, scaler_y, export_dir_epoch)

# %%
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# 计算CNN-KAN模型的性能指标
def calculate_metrics(y_true, y_pred):
    yt = _ensure_2d_numpy(y_true)
    yp = _ensure_2d_numpy(y_pred)
    n_targets = int(yt.shape[1])

    r2_each = []
    rmse_each = []
    mae_each = []
    mape_each = []
    ev_each = []

    for t in range(n_targets):
        ytt = yt[:, t]
        ypt = yp[:, t]
        mae = mean_absolute_error(ytt, ypt)
        mse = mean_squared_error(ytt, ypt)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(ytt, ypt)

        denom = np.where(np.abs(ytt) < 1e-12, 1e-12, ytt)
        mape = float(np.mean(np.abs((ytt - ypt) / denom)) * 100)

        var_y = float(np.var(ytt))
        ev = 1 - float(np.var(ytt - ypt)) / (var_y if var_y > 1e-12 else 1e-12)

        r2_each.append(float(r2))
        rmse_each.append(float(rmse))
        mae_each.append(float(mae))
        mape_each.append(float(mape))
        ev_each.append(float(ev))

    return (
        float(np.mean(r2_each)),
        float(np.mean(rmse_each)),
        float(np.mean(mae_each)),
        float(np.mean(mape_each)),
        float(np.mean(ev_each)),
    )

# 获取训练集和测试集的预测结果
train_observed, train_predicted = get_predictions(model, train_dataloader, device, scaler_y)
test_observed, test_predicted = get_predictions(model, test_dataloader, device, scaler_y)

# 计算评估指标
r2_train = r2_score(train_observed, train_predicted)
rmse_train = np.sqrt(mean_squared_error(train_observed, train_predicted))
r2_test = r2_score(test_observed, test_predicted)
rmse_test = np.sqrt(mean_squared_error(test_observed, test_predicted))

# 使用实际的测试集数据计算Transformer模型的完整性能指标
r2, rmse, mae, mape, ev = calculate_metrics(test_observed, test_predicted)

final_model_metrics = {
    'R²': r2,
    'RMSE': rmse,
    'MAE': mae,
    'MAPE': mape,
    'EV': ev
}

test_observed_mat = _ensure_2d_numpy(test_observed)
test_predicted_mat = _ensure_2d_numpy(test_predicted)
n_targets_radar = int(test_observed_mat.shape[1])
final_model_metrics_each = []
if n_targets_radar >= 2:
    for t in range(n_targets_radar):
        r2_t, rmse_t, mae_t, mape_t, ev_t = calculate_metrics(test_observed_mat[:, t], test_predicted_mat[:, t])
        final_model_metrics_each.append({'R²': r2_t, 'RMSE': rmse_t, 'MAE': mae_t, 'MAPE': mape_t, 'EV': ev_t})

print(f"Transformer模型(最终轮次)测试集性能指标：")
print(f"R² = {r2:.6f}")
print(f"RMSE = {rmse:.6f}")
print(f"MAE = {mae:.6f}")
print(f"MAPE = {mape:.6f}%")
print(f"EV = {ev:.6f}")
if n_targets_radar >= 2:
    for t in range(n_targets_radar):
        d = final_model_metrics_each[t]
        print(f"目标{t+1}: R²={d['R²']:.6f}, RMSE={d['RMSE']:.6f}, MAE={d['MAE']:.6f}, MAPE={d['MAPE']:.6f}%, EV={d['EV']:.6f}")

# 雷达图绘制代码
# 字体和样式设置
PRIMARY_FONT_FAMILY = 'Times New Roman'
FALLBACK_CJK_FONT = 'SimHei'
DEFAULT_FONT_COLOR = 'black'

# 指标名称字体设置
METRIC_NAME_FONT_SIZE = 12
METRIC_NAME_FONT_WEIGHT = 'bold'
METRIC_LABEL_DISTANCE_FACTOR = 0.0075

# 数据点数值标注字体设置
VALUE_ANNOTATION_FONT_SIZE = 10
VALUE_ANNOTATION_FONT_WEIGHT = 'normal'

# 图表标题
TITLE_FONT_SIZE = 16
TITLE_FONT_WEIGHT = 'bold'

# 图例
LEGEND_TITLE_FONT_SIZE = 11
LEGEND_TITLE_FONT_WEIGHT = 'bold'
LEGEND_ITEM_FONT_SIZE = 10

# 线宽设置
GRID_LINE_WIDTH = 0.8
AXIS_LINE_WIDTH = 0.7

# 字体配置
try:
    plt.rcParams['font.family'] = [PRIMARY_FONT_FAMILY, FALLBACK_CJK_FONT]
except RuntimeError:
    print(f"警告: 字体 '{PRIMARY_FONT_FAMILY}' 或 '{FALLBACK_CJK_FONT}' 未完全配置, 可能影响显示效果。")
    plt.rcParams['font.family'] = [FALLBACK_CJK_FONT]
plt.rcParams['axes.unicode_minus'] = False

# 数据定义
labels = np.array(['R²', 'RMSE', 'MAE', 'MAPE', 'EV'])
num_vars = len(labels)

if n_targets_radar >= 2:
    data_models = {
        f"Transformer (t{t+1})": [
            final_model_metrics_each[t]['R²'],
            final_model_metrics_each[t]['RMSE'],
            final_model_metrics_each[t]['MAE'],
            final_model_metrics_each[t]['MAPE'],
            final_model_metrics_each[t]['EV'],
        ]
        for t in range(n_targets_radar)
    }
    palette_radar = ['#e74c3c', '#3498db', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
    markers_radar = ['*', 'o', 's', '^', 'D', 'P']
    model_styles = {
        name: {
            'color': palette_radar[i % len(palette_radar)],
            'marker': markers_radar[i % len(markers_radar)],
            'linewidth': 2.2,
            'markersize': 7,
        }
        for i, name in enumerate(data_models.keys())
    }
    model_order = list(data_models.keys())
    rmse_max = max(d['RMSE'] for d in final_model_metrics_each)
    mae_max = max(d['MAE'] for d in final_model_metrics_each)
    mape_max = max(d['MAPE'] for d in final_model_metrics_each)
    axis_scales = {
        'R²': (0, 1.0),
        'RMSE': (0, (rmse_max if rmse_max > 0 else 1.0) * 1.2),
        'MAE': (0, (mae_max if mae_max > 0 else 1.0) * 1.2),
        'MAPE': (0, (mape_max if mape_max > 0 else 1.0) * 1.5),
        'EV': (0, 1.0)
    }
else:
    data_models = {
        'Transformer (Final)': [
            final_model_metrics['R²'],
            final_model_metrics['RMSE'],
            final_model_metrics['MAE'],
            final_model_metrics['MAPE'],
            final_model_metrics['EV'],
        ]
    }
    model_styles = {
        'Transformer (Final)': {'color': 'red', 'marker': '*', 'linewidth': 2.5, 'markersize': 8}
    }
    model_order = ['Transformer (Final)']
    axis_scales = {
        'R²': (0, 1.0),
        'RMSE': (0, (final_model_metrics['RMSE'] if final_model_metrics['RMSE'] > 0 else 1.0) * 1.2),
        'MAE': (0, (final_model_metrics['MAE'] if final_model_metrics['MAE'] > 0 else 1.0) * 1.2),
        'MAPE': (0, (final_model_metrics['MAPE'] if final_model_metrics['MAPE'] > 0 else 1.0) * 1.5),
        'EV': (0, 1.0)
    }

# 图表参数
center_x, center_y = 0.5, 0.5
chart_scale_factor = 0.33
value_annotation_offset_factor = 0.035
num_grid_levels = 4

# 创建雷达图
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# 角度设置
initial_angle_offset = pi / 2
base_angles_rad = np.linspace(0, 2 * pi, num_vars, endpoint=False)
rotated_angles_rad = [(angle + initial_angle_offset) % (2 * pi) for angle in base_angles_rad]

# 绘制网格和轴
normalized_grid_positions = np.linspace(0, 1, num_grid_levels + 1)[1:]

for norm_pos_idx, norm_pos in enumerate(normalized_grid_positions):
    current_radius_scaled = norm_pos * chart_scale_factor
    grid_points_xy = []
    
    for angle_idx, angle_rad in enumerate(rotated_angles_rad):
        metric_name = labels[angle_idx]
        x_coord = center_x + current_radius_scaled * cos(angle_rad)
        y_coord = center_y + current_radius_scaled * sin(angle_rad)
        grid_points_xy.append((x_coord, y_coord))
        
        if norm_pos_idx == 0:
            # 绘制轴线
            outermost_visual_radius = chart_scale_factor
            axis_end_x = center_x + outermost_visual_radius * cos(angle_rad)
            axis_end_y = center_y + outermost_visual_radius * sin(angle_rad)
            ax.plot([center_x, axis_end_x], [center_y, axis_end_y], color='darkgrey',
                    linestyle='-', linewidth=AXIS_LINE_WIDTH, zorder=1)
            
            # 绘制指标标签
            metric_label_radius = chart_scale_factor + METRIC_LABEL_DISTANCE_FACTOR
            label_x = center_x + metric_label_radius * cos(angle_rad)
            label_y = center_y + metric_label_radius * sin(angle_rad)
            
            epsilon = 1e-3
            ha = 'center'
            va = 'center'
            cos_a = cos(angle_rad)
            sin_a = sin(angle_rad)
            
            if abs(cos_a) < epsilon:
                ha = 'center'
                va = 'bottom' if sin_a > 0 else 'top'
            elif abs(sin_a) < epsilon:
                va = 'center'
                ha = 'left' if cos_a > 0 else 'right'
            else:
                if cos_a > 0 and sin_a > 0: ha = 'left'; va = 'bottom'
                elif cos_a < 0 and sin_a > 0: ha = 'right'; va = 'bottom'
                elif cos_a < 0 and sin_a < 0: ha = 'right'; va = 'top'
                elif cos_a > 0 and sin_a < 0: ha = 'left'; va = 'top'
            
            ax.text(label_x, label_y, metric_name, ha=ha, va=va,
                    fontsize=METRIC_NAME_FONT_SIZE, fontweight=METRIC_NAME_FONT_WEIGHT,
                    color=DEFAULT_FONT_COLOR, zorder=3)
    
    # 绘制网格线
    grid_points_closed_xy = grid_points_xy + [grid_points_xy[0]]
    xs_grid, ys_grid = zip(*grid_points_closed_xy)
    ax.plot(xs_grid, ys_grid, color='grey', linestyle='--', linewidth=GRID_LINE_WIDTH, zorder=1)

for model_name in model_order:
    original_values = data_models[model_name]
    style = model_styles[model_name]
    data_points_xy = []

    for angle_idx, original_value in enumerate(original_values):
        metric_name = labels[angle_idx]
        min_val, max_val = axis_scales[metric_name]
        normalized_value = (original_value - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0.5
        normalized_value_clipped = np.clip(normalized_value, 0, 1.1)

        current_data_visual_radius = normalized_value_clipped * chart_scale_factor
        x_coord = center_x + current_data_visual_radius * cos(rotated_angles_rad[angle_idx])
        y_coord = center_y + current_data_visual_radius * sin(rotated_angles_rad[angle_idx])
        data_points_xy.append((x_coord, y_coord))

        annotation_radius = current_data_visual_radius + value_annotation_offset_factor * chart_scale_factor
        if metric_name in {"R²", "EV"}:
            annotation_radius += 0.01 * chart_scale_factor

        ann_x = center_x + annotation_radius * cos(rotated_angles_rad[angle_idx])
        ann_y = center_y + annotation_radius * sin(rotated_angles_rad[angle_idx])

        if metric_name == "R²":
            ann_x += 0.22 * chart_scale_factor
        elif metric_name == "EV":
            ann_y -= 0.08 * chart_scale_factor

        ax.plot([x_coord, ann_x], [y_coord, ann_y], color=style['color'], linewidth=1, alpha=0.6, zorder=2)

        ha_ann = 'center'
        va_ann = 'center'
        angle_deg = rotated_angles_rad[angle_idx] * 180 / pi

        if 45 <= angle_deg < 135:
            va_ann = 'bottom'
        elif 135 <= angle_deg < 225:
            ha_ann = 'right'
        elif 225 <= angle_deg < 315:
            va_ann = 'top'
        else:
            ha_ann = 'left'

        ax.text(
            ann_x, ann_y, f"{float(original_value):.3f}", ha=ha_ann, va=va_ann,
            fontsize=VALUE_ANNOTATION_FONT_SIZE, fontweight=VALUE_ANNOTATION_FONT_WEIGHT,
            color=style['color'], zorder=4,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=style['color'])
        )

    data_points_closed_xy = data_points_xy + [data_points_xy[0]]
    xs_data, ys_data = zip(*data_points_closed_xy)
    ax.plot(
        xs_data, ys_data, color=style['color'], linewidth=style['linewidth'],
        marker=style['marker'], markersize=style['markersize'], label=model_name, zorder=2
    )
    ax.fill(xs_data, ys_data, color=style['color'], alpha=0.15, zorder=1.5)

handles, labels_legend = ax.get_legend_handles_labels()
legend = fig.legend(
    handles=handles,
    labels=labels_legend,
    loc='center left', bbox_to_anchor=(0.01, 0.5),
    fontsize=LEGEND_ITEM_FONT_SIZE, frameon=True,
    title="模型名称", title_fontsize=LEGEND_TITLE_FONT_SIZE
)

legend.get_title().set_fontsize(LEGEND_TITLE_FONT_SIZE)
legend.get_title().set_fontweight(LEGEND_TITLE_FONT_WEIGHT)

# 添加标题
plt.suptitle('Transformer模型性能雷达图', fontsize=TITLE_FONT_SIZE, fontweight=TITLE_FONT_WEIGHT, y=0.95)

# 调整布局
fig.subplots_adjust(left=-0.1, right=0.95, top=0.93, bottom=0.05)
plt.show()

# %%
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.family'] = ['SimSun']

plt.rcParams['font.size'] = 12  # 设置字体大小
plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签字体大小
plt.rcParams['axes.titlesize'] = 16  # 标题字体大小

# 获取训练集和测试集的预测值和真实值
def get_predictions(model, dataloader, device, scaler_y):
    y_true, y_pred = collect_predictions_matrix(model, dataloader, device, scaler_y)
    if y_true.ndim == 2 and int(y_true.shape[1]) == 1:
        return y_true[:, 0], y_pred[:, 0]
    return y_true, y_pred

# 高级可视化函数
def plot_advanced_regression_results(model, train_loader, test_loader, device, scaler_y):
    train_observed, train_predicted = get_predictions(model, train_loader, device, scaler_y)
    test_observed, test_predicted = get_predictions(model, test_loader, device, scaler_y)

    def _plot_single(train_obs, train_pred, test_obs, test_pred, title, out_pdf):
        r2_train = r2_score(train_obs, train_pred)
        rmse_train = np.sqrt(mean_squared_error(train_obs, train_pred))
        r2_test = r2_score(test_obs, test_pred)
        rmse_test = np.sqrt(mean_squared_error(test_obs, test_pred))

        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])

        ax_main = fig.add_subplot(gs[0, 0])
        ax_residuals = fig.add_subplot(gs[1, 0], sharex=ax_main)
        ax_right_kde = fig.add_subplot(gs[0, 1], sharey=ax_main)

        fig.patch.set_facecolor('#f8f9fa')
        for ax in [ax_main, ax_residuals, ax_right_kde]:
            ax.set_facecolor('#f8f9fa')

        sns.scatterplot(x=train_obs, y=train_pred, color=palette[0], label='训练集数据', alpha=0.7, s=70, ax=ax_main)
        sns.scatterplot(x=test_obs, y=test_pred, color=palette[3], label='测试集数据', alpha=0.7, s=70, ax=ax_main)

        xlim = ax_main.get_xlim()
        x_line = np.linspace(*xlim, 100)
        y_line = x_line
        ax_main.plot(x_line, y_line, 'k-', linewidth=2, label='完美拟合线')

        stats_text = (
            f'R$^2$（训练集）: {r2_train:.3f}\n'
            f'RMSE (训练集): {rmse_train:.3f}\n'
            f'R$^2$（测试集）: {r2_test:.3f}\n'
            f'RMSE (测试集): {rmse_test:.3f}'
        )
        ax_main.text(
            0.05, 0.95, stats_text,
            transform=ax_main.transAxes,
            verticalalignment='top',
            fontsize=14,
            bbox=dict(facecolor='none', edgecolor='none') # 移除统计信息文本框背景
        )

        ax_main.set_xlabel('观测值', fontsize=14, labelpad=10)
        ax_main.set_ylabel('预测值', fontsize=14, labelpad=10)
        ax_main.legend(fontsize=11, frameon=False) # 移除图例背景
        ax_main.tick_params(axis='both', which='major', labelsize=12)

        sns.kdeplot(y=train_obs, color=palette[0], fill=True, label='训练集观测值', ax=ax_right_kde, alpha=0.5)
        sns.kdeplot(y=train_pred, color=palette[0], fill=True, label='训练集预测值', ax=ax_right_kde, alpha=0.5, linestyle='--')
        sns.kdeplot(y=test_obs, color=palette[3], fill=True, label='测试集观测值', ax=ax_right_kde, alpha=0.5)
        sns.kdeplot(y=test_pred, color=palette[3], fill=True, label='测试集预测值', ax=ax_right_kde, alpha=0.5, linestyle='--')

        for spine in ax_right_kde.spines.values():
            spine.set_visible(False)
        ax_right_kde.set_xticks([])
        ax_right_kde.tick_params(axis='y', left=False, labelleft=False)
        ax_right_kde.margins(x=0)
        ax_right_kde.legend(fontsize=10, loc='upper right', frameon=False) # 移除图例背景

        residuals_train = train_pred - train_obs
        residuals_test = test_pred - test_obs

        sns.scatterplot(x=train_obs, y=residuals_train, color=palette[0], alpha=0.7, s=70, ax=ax_residuals, legend=False)
        sns.scatterplot(x=test_obs, y=residuals_test, color=palette[3], alpha=0.7, s=70, ax=ax_residuals, legend=False)
        ax_residuals.axhline(0, color='black', linestyle='--', alpha=0.7, linewidth=2)

        ax_residuals.set_xlabel('观测值', fontsize=14, labelpad=10)
        ax_residuals.set_ylabel('残差', fontsize=14, labelpad=10)
        ax_residuals.tick_params(axis='both', which='major', labelsize=12)

        ax_main.grid(False)
        ax_residuals.grid(False)
        ax_right_kde.grid(False)

        plt.subplots_adjust(wspace=0.05, hspace=0.1)
        ax_right_kde.set_xlabel('')

        plt.suptitle(title, fontsize=18, y=0.98, fontweight='bold')
        if out_pdf:
            plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
        plt.show()

    if isinstance(train_observed, np.ndarray) and train_observed.ndim == 2:
        n_targets = int(train_observed.shape[1])
        for t in range(n_targets):
            _plot_single(
                train_observed[:, t],
                train_predicted[:, t],
                test_observed[:, t],
                test_predicted[:, t],
                f'Transformer回归模型预测性能分析 (目标{t+1})',
                f'Transformer回归预测分析_t{t+1}.pdf',
            )
        return

    _plot_single(
        train_observed,
        train_predicted,
        test_observed,
        test_predicted,
        'Transformer回归模型预测性能分析',
        'Transformer回归预测分析.pdf',
    )

# 使用函数绘制图形
plot_advanced_regression_results(best_model, train_dataloader, test_dataloader, device, scaler_y)

# %%
# 绘制训练过程曲线
epochs_range = range(len(train_loss))

# --- 数据导出逻辑开始 ---
import os
import pandas as pd
# 定义导出目录
export_dir = r'D:\anaconda\transformer\642989960506957Transformer回归加shap(1)\Transformer回归加shap\shuju'
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# 创建包含训练和验证指标的 DataFrame
df_metrics = pd.DataFrame({
    'epoch': list(range(1, len(train_loss) + 1)),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'train_r2': train_r2,
    'val_r2': val_r2
})

# 导出为 CSV 文件
metrics_csv_path = os.path.join(export_dir, 'training_validation_metrics.csv')
df_metrics.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
print(f"训练过程数据已成功导出至: {metrics_csv_path}")
# --- 数据导出逻辑结束 ---

# 设置图形样式
plt.figure(figsize=(12, 5))

# 第一个子图：损失曲线
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss, label='training', linewidth=2, color='#1f77b4')
plt.plot(epochs_range, val_loss, label='validation', linewidth=2, color='#ff7f0e')
plt.xlabel('epoch', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.legend(fontsize=10, frameon=True)
plt.grid(True, alpha=0.3)
# 设置坐标轴样式
ax1 = plt.gca()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(1)
ax1.spines['bottom'].set_linewidth(1)

# 第二个子图：R²曲线（如果有的话，可以替换为其他指标）
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_r2, label='training', linewidth=2, color='#1f77b4')
plt.plot(epochs_range, val_r2, label='validation', linewidth=2, color='#ff7f0e')
plt.xlabel('epoch', fontsize=12)
plt.ylabel('R²', fontsize=12)
plt.legend(fontsize=10, frameon=True)
plt.grid(True, alpha=0.3)
# 设置坐标轴样式
ax2 = plt.gca()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(1)
ax2.spines['bottom'].set_linewidth(1)

# 调整布局
plt.tight_layout()
plt.show()

# %% 验证压力补偿效果
# ==========================================
# 压力补偿验证逻辑：同一浓度不同压力下的预测稳定性
# ==========================================
print("\n--- 开始验证模型压力补偿效果 ---")

def verify_pressure_compensation(model, X_test, scaler_X_pressure, scaler_y, device, export_dir, target_values=None):
    model.eval()
    pressure_points = np.array([1.0, 1.5, 2.0, 2.5, 3.0], dtype=float)
    pressure_scaled = scaler_X_pressure.transform(pressure_points.reshape(-1, 1)).reshape(-1)

    y_test_target = np.asarray(y_test_raw)[:, 0].reshape(-1)
    if target_values is None:
        env_vals = os.getenv("PRESSURE_COMP_TARGETS", "").strip()
        if len(env_vals) > 0:
            target_values = [float(v.strip()) for v in env_vals.split(",") if len(v.strip()) > 0]
        else:
            uniq = np.unique(np.round(y_test_target, 6))
            if uniq.size <= 6:
                target_values = uniq.tolist()
            else:
                qv = np.quantile(uniq, [0.2, 0.5, 0.8])
                target_values = [float(v) for v in qv]

    target_values = [float(v) for v in target_values]
    records = []
    plt.figure(figsize=(10, 6), dpi=600)

    for tval in target_values:
        idx = int(np.argmin(np.abs(y_test_target - tval)))
        true_val = float(y_test_target[idx])
        sample_x = X_test[idx].copy()
        spectra_part = sample_x[:-1]

        preds_raw = []
        for ps in pressure_scaled:
            test_input = np.append(spectra_part, float(ps)).reshape(1, -1)
            with torch.no_grad():
                pred_scaled = model(torch.FloatTensor(test_input).to(device)).detach().cpu().numpy()
            pred_raw = scaler_y.inverse_transform(np.asarray(pred_scaled).reshape(1, -1))[0, 0]
            preds_raw.append(float(pred_raw))

        preds_raw = np.asarray(preds_raw, dtype=float)
        re_pct = np.abs(preds_raw - true_val) / (abs(true_val) if abs(true_val) > 1e-12 else 1e-12) * 100.0

        plt.plot(
            pressure_points,
            preds_raw,
            marker='o',
            markersize=6,
            linewidth=2.2,
            label=f'True≈{true_val:.1f} ppm'
        )
        plt.axhline(true_val, linestyle='--', linewidth=1.2, alpha=0.55, color='gray')

        for p, pr, re in zip(pressure_points, preds_raw, re_pct):
            records.append({
                "target_true_ppm": float(true_val),
                "pressure_atm": float(p),
                "pred_ppm": float(pr),
                "abs_error_ppm": float(abs(pr - true_val)),
                "re_percent": float(re),
            })

    plt.xlabel('Pressure (atm)', fontsize=13, fontweight='bold')
    plt.ylabel('Predicted concentration (ppm)', fontsize=13, fontweight='bold')
    plt.title('Pressure compensation at fixed concentration levels', fontsize=14, fontweight='bold')
    plt.xticks(pressure_points)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in', width=1.6, length=5)
    plt.legend(loc='best', frameon=False, fontsize=10)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color('black')
    plt.grid(False)
    plt.tight_layout()

    csv_save_path = os.path.join(export_dir, 'pressure_compensation_discrete_points.csv')
    fig_save_path = os.path.join(export_dir, 'pressure_compensation_discrete_points.png')
    pd.DataFrame(records).to_csv(csv_save_path, index=False, encoding='utf-8-sig')
    plt.savefig(fig_save_path, bbox_inches='tight', dpi=600)
    plt.show()
    print(f"✅ 离散压力点验证数据已导出至: {csv_save_path}")
    print(f"✅ 离散压力点验证图已保存至: {fig_save_path}")

# 执行验证
verify_pressure_compensation(best_model, X_test, scaler_X_pressure, scaler_y, device, export_dir_epoch)












# %%
#保存最佳模型到文件中
PATH='./Transformer_model.pth'
torch.save(best_model.state_dict(),PATH)

# %%
# 将训练数据重塑为二维DataFrame，方便后续分析
X_train_ = pd.DataFrame(X_train.reshape(X_train.shape[0], -1), columns=columns)

tensor_X_train = torch.tensor(X_train, dtype=torch.float32).to(device)

# 查看转换后的张量形状
tensor_X_train.shape

# %%
def _predict_from_pooled(best_model, pooled, target_index=0):
    feat = best_model.shared_output_trunk(pooled)
    if not hasattr(best_model, "output_heads"):
        raise AttributeError("模型缺少 output_heads，无法进行多目标输出预测")

    outs = [head(feat) for head in best_model.output_heads]
    if len(outs) == 1:
        out = outs[0]
        return out.view(-1)

    if target_index is None:
        out_mat = torch.cat(outs, dim=1)
        return out_mat.sum(dim=1)

    t = int(target_index)
    if t < 0:
        t = 0
    if t >= len(outs):
        t = len(outs) - 1
    return outs[t].view(-1)


def compute_compressed_token_attributions(best_model, X_tensor, peak1_len, peak2_len, device, batch_size=128, target_index=0):
    """\
    计算 CNN 压缩后的 token 表示（每个 token 的 d_model 维向量）对最终输出的贡献度。

    这里把 token 向量当成“Transformer 子模型”的输入，计算 output 对 token 的梯度，
    用 grad * value 作为每个压缩维度的贡献近似（类似一阶泰勒展开）。
    """

    best_model.eval()
    d_model = int(best_model.d_model)

    feature_names = None
    n_tokens_peak1 = None
    n_tokens_peak2 = None

    all_token_values = []
    all_contrib_values = []
    all_preds = []

    for start in range(0, X_tensor.shape[0], batch_size):
        end = min(X_tensor.shape[0], start + batch_size)
        X_batch = X_tensor[start:end]

        with torch.no_grad():
            peak1 = X_batch[:, :peak1_len]
            peak2 = X_batch[:, peak1_len:-1]
            pressure = X_batch[:, -1:].contiguous()

            tok1 = best_model.peak1_cnn(peak1)
            tok2 = best_model.peak2_cnn(peak2)
            base_tokens = torch.cat([tok1, tok2], dim=1)

            n1 = int(tok1.shape[1])
            n2 = int(tok2.shape[1])
            if feature_names is None:
                n_tokens_peak1 = n1
                n_tokens_peak2 = n2
                names = []
                for t in range(n1):
                    for i in range(d_model):
                        names.append(f"peak1_tok{t}_dim_{i}")
                for t in range(n2):
                    for i in range(d_model):
                        names.append(f"peak2_tok{t}_dim_{i}")
                feature_names = names

            type_ids = torch.cat(
                [
                    torch.zeros(n1, dtype=torch.long, device=device),
                    torch.ones(n2, dtype=torch.long, device=device),
                ],
                dim=0,
            ).unsqueeze(0).expand(base_tokens.shape[0], -1)
            type_emb = best_model.token_type_embeddings(type_ids)

        leaf_tokens = base_tokens.detach().requires_grad_(True)
        tokens_in = leaf_tokens + type_emb
        if getattr(best_model, "pos_encoder", None) is not None:
            tokens_in = tokens_in.transpose(0, 1)
            tokens_in = best_model.pos_encoder(tokens_in)
            tokens_in = tokens_in.transpose(0, 1)
        tokens_in.retain_grad()

        encoded = best_model.transformer_encoder(tokens_in, pressure)
        pooled = encoded.mean(dim=1)
        y_hat = _predict_from_pooled(best_model, pooled, target_index=target_index)
        loss = y_hat.sum()

        best_model.zero_grad(set_to_none=True)
        loss.backward()

        grads = leaf_tokens.grad
        if grads is None:
            grads = tokens_in.grad
        if grads is None:
            print(
                "[ERROR] token attribution grad is None.",
                "batch_range=", (start, end),
                "base_tokens_shape=", tuple(base_tokens.shape),
                "leaf_requires_grad=", bool(leaf_tokens.requires_grad),
                "tokens_in_is_leaf=", bool(tokens_in.is_leaf),
            )
            raise RuntimeError("token attribution failed: gradients are None")

        grads = grads.detach()
        contrib = grads * base_tokens.detach()

        all_preds.append(y_hat.detach().cpu().numpy())
        all_token_values.append(base_tokens.detach().cpu().numpy().reshape(base_tokens.shape[0], -1))
        all_contrib_values.append(contrib.detach().cpu().numpy().reshape(contrib.shape[0], -1))

    token_values = np.concatenate(all_token_values, axis=0)
    contrib_values = np.concatenate(all_contrib_values, axis=0)
    preds = np.concatenate(all_preds, axis=0)

    scores = np.mean(np.abs(contrib_values), axis=0)
    scores = np.asarray(scores, dtype=float)
    return {
        "feature_names": feature_names,
        "token_values": token_values,
        "contrib_values": contrib_values,
        "scores": scores,
        "preds": preds,
        "d_model": d_model,
        "n_tokens_peak1": int(n_tokens_peak1) if n_tokens_peak1 is not None else 0,
        "n_tokens_peak2": int(n_tokens_peak2) if n_tokens_peak2 is not None else 0,
    }


def plot_token_contrib_score_distribution(scores, feature_names, out_pdf, cmap_name="inferno"):
    """\
    压缩维度的重要性分布图：
    - 左：全体压缩维度 score 的直方图 + 阈值线
    - 右：按重要性排序后的散点 + LOWESS 平滑 + 阈值线
    """
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess

    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]

    thr_90 = float(np.quantile(scores, 0.90))
    thr_95 = float(np.quantile(scores, 0.95))

    plt.figure(figsize=(12, 5), dpi=300)

    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(scores, bins=40, color="#4c72b0", alpha=0.75, edgecolor="white")
    ax1.axvline(thr_90, color="#dd8452", linestyle="--", linewidth=2, label=f"P90={thr_90:.3e}")
    ax1.axvline(thr_95, color="#c44e52", linestyle="--", linewidth=2, label=f"P95={thr_95:.3e}")
    ax1.set_xlabel("Mean |grad*value| (per compressed dim)")
    ax1.set_ylabel("Counts")
    ax1.set_title("Compressed-dim importance distribution")
    ax1.legend(frameon=False)
    ax1.grid(False)

    ax2 = plt.subplot(1, 2, 2)
    ranks = np.arange(1, len(sorted_scores) + 1)
    ax2.scatter(ranks, sorted_scores, s=14, alpha=0.65, c=sorted_scores, cmap=cmap_name)
    sm = lowess(sorted_scores, ranks, frac=0.25, return_sorted=True)
    ax2.plot(sm[:, 0], sm[:, 1], color="black", linewidth=2, alpha=0.8, label="LOWESS")
    ax2.axhline(thr_90, color="#dd8452", linestyle="--", linewidth=2)
    ax2.axhline(thr_95, color="#c44e52", linestyle="--", linewidth=2)
    ax2.set_xlabel("Rank (descending)")
    ax2.set_ylabel("Mean |grad*value|")
    ax2.set_title("Ranked importance (with smooth trend)")
    ax2.grid(False)

    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close()

    top_names = [feature_names[i] for i in order[:10]]
    top_vals = sorted_scores[:10]
    print("\nToken压缩维度 Top10 (按 Mean|grad*value| 排序):")
    for n, v in zip(top_names, top_vals):
        print(f"  {n}: {v:.6e}")
    print(f"\n重要性阈值: P90={thr_90:.6e}, P95={thr_95:.6e}")

    top5_idx = order[:5]
    top5_names = [feature_names[i] for i in top5_idx]
    top5_vals = [float(scores[i]) for i in top5_idx]
    np.savez(
        "token_top5_from_terminal.npz",
        names=np.array(top5_names, dtype=object),
        scores=np.array(top5_vals, dtype=float),
        idx=np.array(top5_idx, dtype=int),
    )


def plot_top_token_contrib_dependence(token_values, contrib_values, feature_names, top_k, out_prefix, cmap_name="inferno"):
    """\
    对 Top-K 的压缩维度画“特征值-贡献度”依赖图：
    - 背景直方图：特征值分布
    - 散点：每个样本的贡献度（grad*value）
    - LOWESS：整体趋势
    - 阈值线：|贡献度| 的 P95
    """
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    scores = np.mean(np.abs(contrib_values), axis=0)
    order = np.argsort(scores)[::-1]
    top_idx = order[:top_k]

    for rank, idx in enumerate(top_idx, start=1):
        x = token_values[:, idx]
        y = contrib_values[:, idx]
        thr = float(np.quantile(np.abs(y), 0.95))

        fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
        ax2 = ax1.twinx()
        ax2.patch.set_alpha(0)

        counts, bin_edges = np.histogram(x, bins=35)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        norm_bar = Normalize(vmin=counts.min(), vmax=counts.max())
        try:
            from matplotlib import colormaps

            cmap_bar = colormaps.get_cmap(cmap_name)
        except Exception:
            cmap_bar = cm.get_cmap(cmap_name)
        bar_colors = cmap_bar(norm_bar(counts))
        ax1.bar(bin_centers, counts, width=bin_width * 0.7, align='center', color=bar_colors, alpha=0.55)
        ax1.set_ylabel("Counts")
        ax1.grid(False)

        norm_scatter = Normalize(vmin=np.min(y), vmax=np.max(y))
        sc = ax2.scatter(x, y, c=y, cmap=cmap_name, norm=norm_scatter, s=18, alpha=0.65, zorder=3)
        cbar = plt.colorbar(sc, ax=ax2, shrink=1, aspect=20)
        cbar.set_label("Contribution (grad*value)")

        if len(x) > 5:
            sm = lowess(y, x, frac=0.25, return_sorted=True)
            ax2.plot(sm[:, 0], sm[:, 1], color="black", linewidth=2, alpha=0.8, zorder=4)

        ax2.axhline(0, color="black", linestyle="--", linewidth=1)
        ax2.axhline(thr, color="#c44e52", linestyle="--", linewidth=2, alpha=0.9)
        ax2.axhline(-thr, color="#c44e52", linestyle="--", linewidth=2, alpha=0.9)

        ax1.set_xlabel(feature_names[idx])
        ax2.set_ylabel("Contribution")
        ax1.set_title(f"Top{rank}: {feature_names[idx]} (thr |contrib| P95={thr:.2e})")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_top{rank}_{feature_names[idx]}.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig)


def plot_top_token_contrib_heatmaps(token_values, contrib_values, feature_names, top_k, out_prefix, cmap_name="viridis"):
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess

    scores = np.mean(np.abs(contrib_values), axis=0)
    order = np.argsort(scores)[::-1]
    top_idx = order[:top_k]

    for rank, idx in enumerate(top_idx, start=1):
        name = str(feature_names[int(idx)])
        score_val = float(scores[int(idx)])

        x = np.asarray(token_values[:, int(idx)], dtype=float)
        y = np.asarray(contrib_values[:, int(idx)], dtype=float)

        finite_mask = np.isfinite(x) & np.isfinite(y)
        x = x[finite_mask]
        y = y[finite_mask]
        if x.size < 10:
            continue

        thr = float(np.quantile(np.abs(y), 0.95))

        x_lo, x_hi = float(np.quantile(x, 0.01)), float(np.quantile(x, 0.99))
        y_lo, y_hi = float(np.quantile(y, 0.01)), float(np.quantile(y, 0.99))
        if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_lo == x_hi:
            x_lo, x_hi = float(np.min(x)), float(np.max(x))
        if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_lo == y_hi:
            y_lo, y_hi = float(np.min(y)), float(np.max(y))

        x_edges = np.linspace(x_lo, x_hi, 51)
        y_edges = np.linspace(y_lo, y_hi, 51)
        sum_grid, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges), weights=y)
        cnt_grid, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
        mean_grid = sum_grid / (cnt_grid + 1e-12)
        mean_grid[cnt_grid < 3] = np.nan

        vmax = float(np.nanmax(np.abs(mean_grid))) if np.isfinite(np.nanmax(np.abs(mean_grid))) else float(np.max(np.abs(y)))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300, gridspec_kw={"width_ratios": [1, 2]})
        ax0.hist(x, bins=35, color="#4c72b0", alpha=0.75, edgecolor="white")
        ax0.set_xlabel(name)
        ax0.set_ylabel("Counts")
        ax0.set_title("Value distribution")
        ax0.grid(False)

        extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
        im = ax1.imshow(
            mean_grid.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap_name,
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax1, shrink=1, aspect=20)
        cbar.set_label("Mean contribution (grad*value)")

        if x.size > 20:
            sm = lowess(y, x, frac=0.25, return_sorted=True)
            ax1.plot(sm[:, 0], sm[:, 1], color="black", linewidth=2, alpha=0.85)

        ax1.axhline(0, color="black", linestyle="--", linewidth=1)
        ax1.axhline(thr, color="#c44e52", linestyle="--", linewidth=2, alpha=0.9)
        ax1.axhline(-thr, color="#c44e52", linestyle="--", linewidth=2, alpha=0.9)
        ax1.set_xlabel(name)
        ax1.set_ylabel("Contribution (grad*value)")
        ax1.set_title(f"Top{rank}: {name} | mean|grad*value|={score_val:.3e} | P95(|contrib|)={thr:.2e}")
        ax1.grid(False)

        fig.text(
            0.5,
            -0.02,
            "数据来源：Terminal#346-356 计算结果（按 mean|grad*value| 排序）\n"
            "特征选择：取贡献度排名前5的 token 压缩维度；命名格式 peak[1/2]_tok[编号]_dim_[维度]\n"
            "颜色含义：右图颜色表示该(x,y)区域内的平均贡献(grad*value)；红虚线为 |contrib| 的P95阈值",
            ha="center",
            va="top",
            fontsize=10,
        )

        plt.tight_layout()
        plt.savefig(f"{out_prefix}_top{rank}_{name}.pdf", format="pdf", bbox_inches="tight")
        plt.savefig(f"{out_prefix}_top{rank}_{name}.png", format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_token_cumulative_contrib(scores, feature_names, d_model, n_tokens_peak1, n_tokens_peak2, out_pdf):
    """\
    累积贡献图：按重要性排序后画累积占比，并标注 80%/90% 截断点，同时给出三类token占比。
    """
    import matplotlib.pyplot as plt

    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    total = float(np.sum(sorted_scores)) if float(np.sum(sorted_scores)) != 0 else 1.0
    cum = np.cumsum(sorted_scores) / total

    k80 = int(np.searchsorted(cum, 0.80) + 1)
    k90 = int(np.searchsorted(cum, 0.90) + 1)

    def token_type_from_flat_index(flat_idx):
        if flat_idx < int(n_tokens_peak1) * d_model:
            return "peak1"
        if flat_idx < (int(n_tokens_peak1) + int(n_tokens_peak2)) * d_model:
            return "peak2"
        return "other"

    def composition(top_k):
        top_flat = order[:top_k]
        counts = {"peak1": 0, "peak2": 0, "other": 0}
        for i in top_flat:
            counts[token_type_from_flat_index(int(i))] += 1
        denom = float(top_k) if top_k > 0 else 1.0
        return {k: v / denom for k, v in counts.items()}, counts

    comp80, cnt80 = composition(k80)
    comp90, cnt90 = composition(k90)

    fig = plt.figure(figsize=(12, 5), dpi=300)
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(np.arange(1, len(cum) + 1), cum, color="#4c72b0", linewidth=2)
    ax1.axhline(0.80, color="#dd8452", linestyle="--", linewidth=2)
    ax1.axhline(0.90, color="#c44e52", linestyle="--", linewidth=2)
    ax1.axvline(k80, color="#dd8452", linestyle=":", linewidth=2, label=f"80% -> {k80} dims")
    ax1.axvline(k90, color="#c44e52", linestyle=":", linewidth=2, label=f"90% -> {k90} dims")
    ax1.set_xlabel("#Compressed dims (sorted)")
    ax1.set_ylabel("Cumulative contribution ratio")
    ax1.set_title("Cumulative contribution")
    ax1.set_ylim(0, 1.02)
    ax1.legend(frameon=False)
    ax1.grid(False)

    ax2 = plt.subplot(1, 2, 2)
    labels = ["peak1", "peak2", "other"]
    colors = ["#1f77b4", "#ff7f0e", "#7f7f7f"]
    vals80 = [comp80[l] for l in labels]
    vals90 = [comp90[l] for l in labels]
    x = np.arange(len(labels))
    width = 0.35
    ax2.bar(x - width / 2, vals80, width, label=f"Top{k80} (80%)", color=colors, alpha=0.75)
    ax2.bar(x + width / 2, vals90, width, label=f"Top{k90} (90%)", color=colors, alpha=0.35)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Proportion")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Token composition within cutoff")
    ax2.legend(frameon=False)
    ax2.grid(False)

    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("\n累积贡献截断点:")
    print(f"  80%: top {k80} dims -> counts {cnt80}")
    print(f"  90%: top {k90} dims -> counts {cnt90}")


def plot_token_decision_contour(best_model, token_values, contrib_values, preds, scores, feature_names, d_model, n_tokens_peak1, n_tokens_peak2, out_pdf, device, grid_size=120, target_index=0):
    """\
    在“压缩 token 特征空间”里画 2D 等高线：
    - 选 Top2 的压缩维度作为 x/y
    - 其他维度固定为均值
    - 等高线值为模型输出（scaled output）
    """
    import matplotlib.pyplot as plt

    order = np.argsort(scores)[::-1]
    idx_x = int(order[0])
    idx_y = int(order[1])

    base_mean = np.mean(token_values, axis=0)

    x_vals = token_values[:, idx_x]
    y_vals = token_values[:, idx_y]

    x_min, x_max = float(np.quantile(x_vals, 0.02)), float(np.quantile(x_vals, 0.98))
    y_min, y_max = float(np.quantile(y_vals, 0.02)), float(np.quantile(y_vals, 0.98))

    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    XX, YY = np.meshgrid(xs, ys)

    grid_flat = np.tile(base_mean, (grid_size * grid_size, 1))
    grid_flat[:, idx_x] = XX.reshape(-1)
    grid_flat[:, idx_y] = YY.reshape(-1)

    token_count = int(n_tokens_peak1) + int(n_tokens_peak2)
    if token_count <= 0:
        token_count = int(token_values.shape[1] // d_model)
    grid_tokens = torch.tensor(grid_flat, dtype=torch.float32, device=device).view(-1, token_count, d_model)
    with torch.no_grad():
        type_ids = torch.cat(
            [
                torch.zeros(int(n_tokens_peak1), dtype=torch.long, device=device),
                torch.ones(int(n_tokens_peak2), dtype=torch.long, device=device),
            ],
            dim=0,
        ).unsqueeze(0).expand(grid_tokens.shape[0], -1)
        type_emb = best_model.token_type_embeddings(type_ids)
        tokens_in = grid_tokens + type_emb
        if getattr(best_model, "pos_encoder", None) is not None:
            tokens_in = tokens_in.transpose(0, 1)
            tokens_in = best_model.pos_encoder(tokens_in)
            tokens_in = tokens_in.transpose(0, 1)
        p_fixed = torch.zeros((tokens_in.shape[0], 1), dtype=torch.float32, device=device)
        encoded = best_model.transformer_encoder(tokens_in, p_fixed)
        pooled = encoded.mean(dim=1)
        zz = _predict_from_pooled(best_model, pooled, target_index=target_index).cpu().numpy().reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    cf = ax.contourf(XX, YY, zz, levels=20, cmap="inferno", alpha=0.95)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("Model output (scaled)")

    median_pred = float(np.median(preds))
    ax.contour(XX, YY, zz, levels=[median_pred], colors="white", linewidths=2, linestyles="--")

    sc = ax.scatter(x_vals, y_vals, c=preds, cmap="inferno", s=18, alpha=0.65, edgecolors="none")
    ax.set_xlabel(feature_names[idx_x])
    ax.set_ylabel(feature_names[idx_y])
    ax.set_title("Token compressed space: decision contour (Top2 dims)")
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


print("\n开始计算CNN压缩token特征的贡献度(grad*value)...")
token_attr = compute_compressed_token_attributions(
    best_model=best_model,
    X_tensor=tensor_X_train,
    peak1_len=PEAK1_LEN,
    peak2_len=PEAK2_LEN,
    device=device,
    batch_size=128,
    target_index=int(MODEL_TARGET_INDEX),
)

plot_token_contrib_score_distribution(
    scores=token_attr["scores"],
    feature_names=token_attr["feature_names"],
    out_pdf="token_contrib_score_distribution.pdf",
    cmap_name="inferno",
)

plot_top_token_contrib_dependence(
    token_values=token_attr["token_values"],
    contrib_values=token_attr["contrib_values"],
    feature_names=token_attr["feature_names"],
    top_k=5,
    out_prefix="token_contrib_dependence",
    cmap_name="inferno",
)

plot_top_token_contrib_heatmaps(
    token_values=token_attr["token_values"],
    contrib_values=token_attr["contrib_values"],
    feature_names=token_attr["feature_names"],
    top_k=5,
    out_prefix="token_contrib_heatmap",
    cmap_name="viridis",
)

plot_token_cumulative_contrib(
    scores=token_attr["scores"],
    feature_names=token_attr["feature_names"],
    d_model=token_attr["d_model"],
    n_tokens_peak1=token_attr["n_tokens_peak1"],
    n_tokens_peak2=token_attr["n_tokens_peak2"],
    out_pdf="token_contrib_cumulative.pdf",
)

plot_token_decision_contour(
    best_model=best_model,
    token_values=token_attr["token_values"],
    contrib_values=token_attr["contrib_values"],
    preds=token_attr["preds"],
    scores=token_attr["scores"],
    feature_names=token_attr["feature_names"],
    d_model=token_attr["d_model"],
    n_tokens_peak1=token_attr["n_tokens_peak1"],
    n_tokens_peak2=token_attr["n_tokens_peak2"],
    out_pdf="token_contrib_decision_contour.pdf",
    device=device,
    grid_size=120,
    target_index=int(MODEL_TARGET_INDEX),
)

import shap
# 默认开启 SHAP 图像保存；如需关闭可设置环境变量 SHAP_SAVE_FIGURES=0
SHAP_SAVE_FIGURES = bool(int(os.getenv("SHAP_SAVE_FIGURES", "0")))

#创建 SHAP DeepExplainer
#SHAP 值的计算确实相对较慢，尤其对于深度学习模型（如 CNN）和大规模数据集。这是因为 SHAP 值基于 Shapley 值的原理，需要计算每个输入特征对模型输出的边际贡献，这通常涉及大量模型前向传递操作。
# 计算 SHAP 值
explainer = shap.DeepExplainer(best_model,tensor_X_train[:50,:])#用前50个样本初始化DeepExplainer，这样运行速度会

shap_values = explainer.shap_values(tensor_X_train, check_additivity=False)

# %%
def _normalize_shap_values_to_2d(shap_values, n_features, name="shap_values", target_index=0):
    if isinstance(shap_values, list):
        if target_index is None:
            raise ValueError(f"{name} 为 list 时必须提供 target_index")
        shap_values = shap_values[int(target_index)]
    sv = shap_values
    sv = np.array(sv)
    sv = np.squeeze(sv)

    if sv.ndim == 1:
        if int(sv.size) != int(n_features):
            raise ValueError(f"{name} 维度不符合预期: shape={tuple(sv.shape)}, 期望特征数={int(n_features)}")
        return sv.reshape(1, -1)

    if sv.ndim == 2:
        if int(sv.shape[1]) == int(n_features):
            return sv
        if int(sv.shape[0]) == int(n_features):
            return sv.T
        raise ValueError(f"{name} 维度不符合预期: shape={tuple(sv.shape)}, 期望特征数={int(n_features)}")

    if sv.ndim == 3:
        if int(sv.shape[1]) == int(n_features):
            return sv.mean(axis=2)
        if int(sv.shape[2]) == int(n_features):
            return sv.mean(axis=1)
        raise ValueError(f"{name} 维度不符合预期: shape={tuple(sv.shape)}, 期望特征数={int(n_features)}")

    raise ValueError(f"{name} 维度不支持: shape={tuple(sv.shape)}")


shap_values_2d = _normalize_shap_values_to_2d(shap_values, len(columns), name="shap_values", target_index=int(MODEL_TARGET_INDEX))
raw_sv0 = shap_values[int(MODEL_TARGET_INDEX)] if isinstance(shap_values, list) else shap_values
print(f"SHAP 原始形状: {tuple(np.array(raw_sv0).shape)}")
print(f"SHAP 归一化后形状: {tuple(shap_values_2d.shape)}")

if isinstance(shap_values, list) and len(shap_values) > 1:
    for t in range(len(shap_values)):
        sv2d_t = _normalize_shap_values_to_2d(shap_values, len(columns), name=f"shap_values_t{t+1}", target_index=t)

        mean_abs = np.mean(np.abs(sv2d_t), axis=0)
        order = np.argsort(mean_abs)[::-1]
        top_k = 10
        top_idx = order[:top_k]
        top_features = [columns[int(i)] for i in top_idx]
        top_vals = mean_abs[top_idx]
        plt.figure(figsize=(8, 6), dpi=300)
        y_pos = np.arange(len(top_features))
        ax_top = plt.gca()
        ax_top.barh(y_pos, top_vals, color="#1f77b4", alpha=0.85)
        ax_top.set_yticks(y_pos)
        ax_top.set_yticklabels(top_features)
        ax_top.set_xlabel("Mean |SHAP value|")
        ax_top.set_title(f"Top 10 Features by SHAP Value (目标{t+1})")
        ax_top.invert_yaxis()
        plt.tight_layout()
        if SHAP_SAVE_FIGURES:
            plt.savefig(f"top10_shap_features_t{t+1}.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 8), dpi=300)
        shap.summary_plot(
            sv2d_t,
            X_train_,
            feature_names=columns,
            plot_type="dot",
            show=False,
            color_bar=True,
            cmap="inferno",
            max_display=top_k,
        )
        plt.tight_layout()
        if SHAP_SAVE_FIGURES:
            plt.savefig(f"shap_summary_dot_t{t+1}.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 8), dpi=300)
        shap.summary_plot(
            sv2d_t,
            X_train_,
            feature_names=columns,
            plot_type="bar",
            show=False,
            color="black",
            max_display=top_k,
        )
        plt.tight_layout()
        if SHAP_SAVE_FIGURES:
            plt.savefig(f"shap_summary_bar_t{t+1}.png", dpi=300, bbox_inches="tight")
        plt.close()


def calculate_shap_values(shap_values, X_test, feature_names):
    """
    计算SHAP值并保持原始特征顺序
    """

    sv2d = _normalize_shap_values_to_2d(shap_values, len(feature_names), name="shap_values")
    arr = np.abs(sv2d)
    feature_importance = arr.mean(axis=0)
    feature_importance = np.asarray(feature_importance, dtype=float)
    feature_names_with_shap = [f"{name} ({float(value):.4f})" for name, value in zip(feature_names, feature_importance)]

    return feature_names_with_shap, feature_importance


def group_shap_importance(shap_values, peak1_len, peak2_len):
    total_features = peak1_len + peak2_len + 1
    sv = _normalize_shap_values_to_2d(shap_values, total_features, name="shap_values")
    peak1_idx = np.arange(0, peak1_len)
    peak2_idx = np.arange(peak1_len, peak1_len + peak2_len)
    pressure_idx = np.array([peak1_len + peak2_len])

    def group_score(idx):
        group_sv = sv[:, idx]
        return float(np.mean(np.abs(group_sv)))

    scores = {
        "peak1": group_score(peak1_idx),
        "peak2": group_score(peak2_idx),
        "pressure": group_score(pressure_idx),
    }
    return scores


def grouped_shap_by_pressure_quantiles(shap_values, X_df, peak1_len, peak2_len, low_q=0.2, high_q=0.8):
    total_features = peak1_len + peak2_len + 1
    sv = _normalize_shap_values_to_2d(shap_values, total_features, name="shap_values")

    pressure_col = peak1_len + peak2_len
    pressure_vals = X_df.iloc[:, pressure_col].values
    low_thr = np.quantile(pressure_vals, low_q)
    high_thr = np.quantile(pressure_vals, high_q)

    low_mask = pressure_vals <= low_thr
    high_mask = pressure_vals >= high_thr

    scores_low = group_shap_importance(sv[low_mask], peak1_len, peak2_len)
    scores_high = group_shap_importance(sv[high_mask], peak1_len, peak2_len)
    return scores_low, scores_high

# 计算特征重要性
features_with_shap, importance_values = calculate_shap_values(shap_values, X_test, columns)

global_group_scores = group_shap_importance(shap_values, PEAK1_LEN, PEAK2_LEN)
print("\n按物理分区分组的全体样本SHAP贡献:")
for k, v in global_group_scores.items():
    print(f"  {k}: {v:.6f}")

low_group_scores, high_group_scores = grouped_shap_by_pressure_quantiles(shap_values, X_train_, PEAK1_LEN, PEAK2_LEN, low_q=0.2, high_q=0.8)
print("\n低压力分组(分位数<=20%)的分组SHAP贡献:")
for k, v in low_group_scores.items():
    print(f"  {k}: {v:.6f}")
print("\n高压力分组(分位数>=80%)的分组SHAP贡献:")
for k, v in high_group_scores.items():
    print(f"  {k}: {v:.6f}")

# %%
import matplotlib.pyplot as plt
import shap
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable

# 创建主图（用来画蜂巢图）
fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1200)

# 计算每个特征的平均SHAP值
sv_plot = shap_values[int(MODEL_TARGET_INDEX)] if isinstance(shap_values, list) else shap_values
msv_arr = np.abs(_normalize_shap_values_to_2d(sv_plot, len(columns), name="sv_plot"))
mean_shap_values = msv_arr.mean(axis=0)
total_importance = mean_shap_values.sum()
percentages = (mean_shap_values / (total_importance if total_importance != 0 else 1.0)) * 100

# 创建特征名称和对应的数值
feature_importance_data = list(zip(columns, mean_shap_values, percentages))
# 按重要性排序
feature_importance_data.sort(key=lambda x: x[1], reverse=True)

try:
    _p_idx = int(globals().get("PEAK1_LEN", 0)) + int(globals().get("PEAK2_LEN", 0))
    if 0 <= _p_idx < int(len(columns)):
        _p_name = str(columns[int(_p_idx)])
        _p_val = float(mean_shap_values[int(_p_idx)])
        _rank = int(np.argsort(mean_shap_values)[::-1].tolist().index(int(_p_idx)) + 1)
        _in_top10 = any(str(n) == _p_name for (n, _, _) in feature_importance_data[:10])
        print(f"[SHAP][CHECK] pressure_feature={_p_name} idx={int(_p_idx)} mean|shap|={_p_val:.6f} rank={_rank} top10={bool(_in_top10)} raw_target_col={int(globals().get('RAW_TARGET_COL', 0))} model_target_index={int(globals().get('MODEL_TARGET_INDEX', 0))}")
except Exception as _e:
    print("[SHAP][WARN] 压力特征SHAP排名检查失败:", repr(_e))

top_feature_names_for_dependence = [d[0] for d in feature_importance_data[:5]]

top_k = 10
top_data = feature_importance_data[:top_k]
top_features = [d[0] for d in top_data]
top_importance = np.array([float(d[1]) for d in top_data])
top_values = X_train_[top_features].values
val_stat = np.mean(np.abs(top_values), axis=0)
if val_stat.max() > val_stat.min():
    norm_vals = (val_stat - val_stat.min()) / (val_stat.max() - val_stat.min())
else:
    norm_vals = np.zeros_like(val_stat)
cmap = plt.get_cmap("viridis")
colors = cmap(norm_vals)
plt.figure(figsize=(8, 6), dpi=300)
y_pos = np.arange(len(top_features))
ax_top = plt.gca()
ax_top.barh(y_pos, top_importance, color=colors)
ax_top.set_yticks(y_pos)
ax_top.set_yticklabels(top_features)
ax_top.set_xlabel("Mean |SHAP value|")
ax_top.set_title("Top 10 Features by SHAP Value")
sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=val_stat.min(), vmax=val_stat.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax_top)
cbar.set_label("Average |Feature Value|")
ax_top.invert_yaxis()
plt.tight_layout()
if SHAP_SAVE_FIGURES:
    plt.savefig("top10_shap_features.png", dpi=300)

# 绘制蜂巢图
shap.summary_plot(
    shap_values_2d,
    tensor_X_train,
    columns,
    plot_type="dot",
    show=False,
    color_bar=True,
    cmap='inferno',
    max_display=top_k,
)
plt.gca().set_position([0.18, 0.2, 0.66, 0.65])  # 调整图表位置，使主图与右侧色条更紧凑



ax1 = plt.gca()

fig = plt.gcf()  # 获取当前Figure
ax_cbar = fig.axes[-1]  # 获取最后一个轴（颜色条）


# 创建共享 y 轴的另一个图，绘制特征贡献图在顶部x轴
ax2 = ax1.twiny()
shap.summary_plot(
    shap_values_2d,
    tensor_X_train,
    columns,
    plot_type="bar",
    show=False,
    color='black',
    max_display=top_k,
)
plt.gca().set_position([0.18, 0.2, 0.66, 0.65])  # 调整图表位置，与蜂巢图对齐并保持紧凑

# 调整右侧颜色条与标签距离，避免过远
ax_cbar.set_position([0.85, 0.2, 0.03, 0.65])
ax_cbar.tick_params(labelsize=11, width=1.2)
# 强制右侧颜色条标签与刻度使用 Times New Roman + Bold（显式 FontProperties，避免被覆盖）
_cb_font = FontProperties(family="Times New Roman", weight="bold", size=13)
ax_cbar.set_ylabel("Average |Feature Value|", labelpad=6)
ax_cbar.yaxis.label.set_fontproperties(_cb_font)
for _tl in ax_cbar.get_yticklabels():
    _tl.set_fontproperties(FontProperties(family="Times New Roman", weight="bold", size=11))

# 在顶部 X 轴添加一条横线
y_max = len(columns)
ax2.axhline(y=13, color='gray', linestyle='-', linewidth=1)

# 调整透明度
bars = ax2.patches  # 获取所有的柱状图对象
for bar in bars:
    bar.set_alpha(0.2)  # 设置透明度

# 移动顶部的 X 轴，避免与底部 X 轴重叠
ax2.xaxis.set_label_position('top')  # 将标签移动到顶部
ax2.xaxis.tick_top()  # 将刻度也移动到顶部

ax2.set_xlabel(
    'Mean Shapley Value (Feature Importance)',
    fontsize=12,
    fontname='Times New Roman',
    fontweight='bold'
)
# 在右侧添加平均SHAP值和百分比文本（按重要性排序）
# 获取当前图中特征的顺序（从上到下）
current_y_labels = [tick.get_text() for tick in ax1.get_yticklabels()]

# 为每个位置添加对应的数值和百分比
for i, feature_name in enumerate(current_y_labels):
    # 找到该特征在排序后数据中的信息
    feature_data = next((item for item in feature_importance_data if item[0] == feature_name), None)
    if feature_data:
        feature, mean_val, percentage = feature_data
        y_pos = i  # 当前特征在图中的y位置
        
        # 分别绘制SHAP值（蓝色）和百分比（红色）
        # 绘制SHAP值（蓝色）
        ax1.text(ax1.get_xlim()[1] * 0.93, y_pos + 0.18, f'{mean_val:.4f}',
                 verticalalignment='center', fontsize=9, color='blue', fontname='Times New Roman', fontweight='bold')
        
        # 绘制百分比（红色）
        ax1.text(ax1.get_xlim()[1] * 0.93, y_pos - 0.18, f'{percentage:.3f}%',
                 verticalalignment='center', fontsize=9, color='red', fontname='Times New Roman', fontweight='bold')
# 设置y轴标签
ax1.set_ylabel('Feature', fontsize=12, fontname='Times New Roman', fontweight='bold')

# 全局蜂巢图字体统一：Times New Roman + Bold（主轴/顶部轴/颜色条轴/文本）
for _ax in [ax1, ax2, ax_cbar]:
    _ax_title = _ax.title
    _ax_title.set_fontname('Times New Roman')
    _ax_title.set_fontweight('bold')

    _xlabel = _ax.xaxis.label
    _xlabel.set_fontname('Times New Roman')
    _xlabel.set_fontweight('bold')

    _ylabel = _ax.yaxis.label
    _ylabel.set_fontname('Times New Roman')
    _ylabel.set_fontweight('bold')

    for _tick in _ax.get_xticklabels() + _ax.get_yticklabels():
        _tick.set_fontname('Times New Roman')
        _tick.set_fontweight('bold')

    for _txt in _ax.texts:
        _txt.set_fontname('Times New Roman')
        _txt.set_fontweight('bold')

for _txt in fig.texts:
    _txt.set_fontname('Times New Roman')
    _txt.set_fontweight('bold')

plt.grid(False)
plt.tight_layout() 


# 保存图片
if SHAP_SAVE_FIGURES:
    plt.savefig("SHAP_combined_with_values_and_percentages.png", format='png', bbox_inches='tight', dpi=300)
plt.show()

# 打印特征重要性排序
print("\n特征重要性排序：")
print("特征名\t\t平均SHAP值\t百分比")
print("-" * 40)
for feature, mean_val, percentage in feature_importance_data[:10]:
    print(f"{feature}\t\t{mean_val:.4f}\t\t{percentage:.3f}%")

# %%
shap.summary_plot(
    shap_values_2d,
    X_train_,
    plot_type="bar",
    show=False,
    feature_names=features_with_shap,
    max_display=10,
)
plt.title('SHAP_numpy Sorted Feature Importance')
plt.tight_layout()

# %%
def plot_shap_dependence(feature_name, x_values, shap_values_for_feature, custom_annotation=None):
    """
    绘制并保存单个特征的SHAP依赖图。
    **SHAP值散点图**: 显示每个样本的特征值与其对应的SHAP值的关系（使用bwr颜色映射）。
    **特征值分布直方图**: 以背景条形图的形式展示该特征在数据集中的分布情况（使用bwr颜色映射）。
    **LOWESS平滑拟合曲线**: 揭示SHAP值随特征值变化的平均趋势（深蓝色实线）。
    **置信区间**: 为LOWESS曲线提供统计可靠性范围，通常是95%置信区间（浅蓝色填充区域）。
    **阈值（交点）标定**: 自动寻找并标记出拟合曲线与y=0的交点，这些点是特征影响方向（正/负）发生改变的关键阈值（绿色虚线和标签）。
    **自定义注释**: 允许用户在图上添加自定义文本。
    
    参数说明:
    feature_name (str): 要绘制的特征的名称。将用作图表X轴标签和输出文件名的一部分。
    x_values (pd.Series or np.array): 该特征在数据集中的所有样本值。
    shap_values_for_feature (np.array): 与x_values一一对应的SHAP值。
    custom_annotation (dict, optional): 一个可选字典，用于在图上添加自定义注释。
                                     例如: {'text': '关键区域', 'x': 0.8, 'y': 0.8}
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    print(f"  -> 正在绘制特征: {feature_name} ...")  # 打印正在绘制哪个特征的提示

    x_values = np.asarray(x_values, dtype=float).reshape(-1)
    shap_values_for_feature = np.asarray(shap_values_for_feature, dtype=float).reshape(-1)

    def _resolve_feature_idx(_feature_name):
        _cols = globals().get("columns", None)
        if _cols is None:
            return None
        try:
            _cols_list = _cols.tolist() if hasattr(_cols, "tolist") else list(_cols)
            return int(_cols_list.index(_feature_name))
        except Exception:
            return None

    feature_idx = _resolve_feature_idx(feature_name)

    x_raw = x_values
    try:
        n_features = int(len(columns))
        if bool(globals().get("USE_SPLIT_SCALING", False)):
            _p_idx = int(globals().get("PRESSURE_INDEX", -1))
            if _p_idx < 0:
                p_idx = int(n_features + _p_idx)
            else:
                p_idx = int(_p_idx)
            p_idx = max(0, min(p_idx, n_features - 1))

            if feature_idx is not None and int(feature_idx) == int(p_idx):
                sp = globals().get("scaler_X_pressure", None)
                if sp is not None and hasattr(sp, "scale_"):
                    _sc = float(np.asarray(sp.scale_).reshape(-1)[0])
                    if not np.isfinite(_sc) or float(_sc) == 0.0:
                        _sc = 1.0
                    if hasattr(sp, "mean_"):
                        _mu = float(np.asarray(sp.mean_).reshape(-1)[0])
                        x_raw = x_values * _sc + _mu
                    elif hasattr(sp, "min_"):
                        _mn = float(np.asarray(sp.min_).reshape(-1)[0])
                        x_raw = (x_values - _mn) / _sc
            elif feature_idx is not None and int(feature_idx) < int(p_idx):
                sx = globals().get("scaler_X_spectra", None)
                if sx is not None and hasattr(sx, "scale_"):
                    _sc = float(np.asarray(sx.scale_).reshape(-1)[int(feature_idx)])
                    if not np.isfinite(_sc) or float(_sc) == 0.0:
                        _sc = 1.0
                    if hasattr(sx, "mean_"):
                        _mu = float(np.asarray(sx.mean_).reshape(-1)[int(feature_idx)])
                        x_raw = x_values * _sc + _mu
                    elif hasattr(sx, "min_"):
                        _mn = float(np.asarray(sx.min_).reshape(-1)[int(feature_idx)])
                        x_raw = (x_values - _mn) / _sc
            else:
                if globals().get("scaler_X", None) is not None and hasattr(scaler_X, "scale_") and feature_idx is not None:
                    _sc = float(np.asarray(scaler_X.scale_).reshape(-1)[int(feature_idx)])
                    if not np.isfinite(_sc) or float(_sc) == 0.0:
                        _sc = 1.0
                    if hasattr(scaler_X, "mean_"):
                        _mu = float(np.asarray(scaler_X.mean_).reshape(-1)[int(feature_idx)])
                        x_raw = x_values * _sc + _mu
                    elif hasattr(scaler_X, "min_"):
                        _mn = float(np.asarray(scaler_X.min_).reshape(-1)[int(feature_idx)])
                        x_raw = (x_values - _mn) / _sc
        else:
            if globals().get("scaler_X", None) is not None and hasattr(scaler_X, "scale_") and feature_idx is not None:
                _sc = float(np.asarray(scaler_X.scale_).reshape(-1)[int(feature_idx)])
                if not np.isfinite(_sc) or float(_sc) == 0.0:
                    _sc = 1.0
                if hasattr(scaler_X, "mean_"):
                    _mu = float(np.asarray(scaler_X.mean_).reshape(-1)[int(feature_idx)])
                    x_raw = x_values * _sc + _mu
                elif hasattr(scaler_X, "min_"):
                    _mn = float(np.asarray(scaler_X.min_).reshape(-1)[int(feature_idx)])
                    x_raw = (x_values - _mn) / _sc
    except Exception:
        x_raw = x_values

    shap_raw = shap_values_for_feature
    try:
        sy = globals().get("scaler_y", None)
        if sy is not None and hasattr(sy, "scale_"):
            _t = int(globals().get("MODEL_TARGET_INDEX", 0))
            _scale_arr = np.asarray(sy.scale_).reshape(-1)
            _t = max(0, min(int(_t), int(_scale_arr.size) - 1))
            _sc = float(_scale_arr[_t])
            if not np.isfinite(_sc) or float(_sc) == 0.0:
                _sc = 1.0
            if hasattr(sy, "mean_"):
                shap_raw = shap_values_for_feature * _sc
            elif hasattr(sy, "min_"):
                shap_raw = shap_values_for_feature / _sc
    except Exception:
        shap_raw = shap_values_for_feature
    fig_dep, ax1 = plt.subplots(figsize=(9.6, 6.9), dpi=180)  # 增大画布与分辨率，提升PPT清晰度
    ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
    ax2.patch.set_alpha(0)  # 将第二个y轴的背景设置为透明
    dep_font = "Times New Roman"
    dep_label_fs = 22  # 在当前基础上再提高3号
    dep_tick_fs = 20   # 在当前基础上再提高3号
    
    # 计算特征值的分布直方图数据
    counts, bin_edges = np.histogram(x_raw, bins=30)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # 计算每个柱子的中心位置
    bin_width = bin_edges[1] - bin_edges[0]  # 计算每个柱子的宽度
    
    # 为直方图创建颜色映射（使用偏深色段，提升投影/屏摄可读性）
    norm_bar = Normalize(vmin=counts.min(), vmax=counts.max())
    cmap_bar = plt.get_cmap('magma')
    bar_colors = cmap_bar(0.18 + 0.62 * norm_bar(counts))
    
    # 绘制带颜色映射的直方图
    bars = ax1.bar(
        bin_centers,
        counts,
        width=bin_width * 0.85,
        align='center',
        color=bar_colors,
        edgecolor='#3a2400',
        linewidth=1.3,
        alpha=0.92,
        label='Distribution'
    )
    
    ax1.set_ylabel('Counts', fontsize=dep_label_fs, fontname=dep_font, fontweight='bold')  # 设置ax1的y轴标签
    ax1.set_ylim(0, counts.max() * 1.1)  # 设置ax1的y轴范围
    
    # 为散点图创建颜色映射（基于SHAP值）
    norm_scatter = Normalize(vmin=shap_raw.min(), vmax=shap_raw.max())
    cmap_scatter = plt.get_cmap('magma')
    
    # 绘制带颜色映射的散点图
    scatter = ax2.scatter(
        x_raw, shap_raw,
        c=shap_raw, cmap='magma',
        alpha=0.85, s=38, label='Sample', zorder=2
    )
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax2, shrink=1, aspect=24, pad=0.08)
    cbar.set_label('SHAP Values', fontsize=dep_label_fs, fontname=dep_font, fontweight='bold')
    cbar.ax.tick_params(labelsize=dep_tick_fs, width=1.6, length=5)
    for _tl in cbar.ax.get_yticklabels():
        _tl.set_fontname(dep_font)
        _tl.set_fontweight('bold')
    
    if len(x_raw) > 1:  # 检查样本数量是否足够
        main_fit, ci_data = bootstrap_lowess_ci(x_raw, shap_raw, frac=0.3)  # 计算lowess平滑曲线和置信区间
        if main_fit is not None and ci_data is not None:  # 如果成功计算
            ax2.plot(main_fit[:, 0], main_fit[:, 1], color='#222222', lw=2.8, label='LOWESS Fit', zorder=4, alpha=0.9)  # 绘制主平滑曲线
            ax2.fill_between(ci_data[0], ci_data[1], ci_data[2], color='#7f7f7f', alpha=0.14, label='95%CI')  # 填充置信区间
            ax2.axhline(0, color='#1f1f1f', linestyle='--', lw=1.8, zorder=1)  # 绘制y=0的参考线
            find_and_plot_crossings(ax2, main_fit[:, 0], main_fit[:, 1], 'black')  # 寻找并绘制阈值线
    
    # ax2.set_ylabel('SHAP VALUES', fontsize=12)  # 设置ax2的y轴标签
    y_max = np.abs(shap_raw).max() * 1.15  # 计算ax2的y轴范围
    if y_max < 1e-6: y_max = 1  # 避免范围过小
    ax2.set_ylim(-y_max, y_max)  # 设置ax2的y轴范围
    ax1.set_xlabel(f'{feature_name}', fontsize=dep_label_fs, fontname=dep_font, fontweight='bold')  # 设置共享的x轴标签
    ax1.tick_params(axis='both', which='major', labelsize=dep_tick_fs, width=1.8, length=6, direction='in')
    ax2.tick_params(axis='both', which='major', labelsize=dep_tick_fs, width=1.8, length=6, direction='in')
    for _sp in ax1.spines.values():
        _sp.set_linewidth(1.8)
    for _sp in ax2.spines.values():
        _sp.set_linewidth(1.6)
    for _tick in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
        _tick.set_fontname(dep_font)
        _tick.set_fontweight('bold')
    
    if custom_annotation and isinstance(custom_annotation, dict):  # 检查是否有自定义注释
        text = custom_annotation.get('text', '')  # 获取注释文本
        x_pos = custom_annotation.get('x', 0.95)  # 获取注释x坐标
        y_pos = custom_annotation.get('y', 0.95)  # 获取注释y坐标
        props = {'ha': custom_annotation.get('ha', 'right'), 'va': custom_annotation.get('va', 'top'),
                'fontsize': custom_annotation.get('fontsize', dep_label_fs), 'color': custom_annotation.get('color', 'darkred'),
                 'fontweight': custom_annotation.get('fontweight', 'bold')}  # 定义注释的样式
        ax1.text(x_pos, y_pos, text, transform=ax1.transAxes, **props)  # 在图上添加自定义注释
    
    h1, l1 = ax1.get_legend_handles_labels()  # 获取ax1的图例句柄和标签
    h2, l2 = ax2.get_legend_handles_labels()  # 获取ax2的图例句柄和标签
    # ax2.legend(h2 + h1, l2 + l1, loc='upper right', fontsize=10)  # 合并两个y轴的图例并显示
    
    fig_dep.subplots_adjust(left=0.11, right=0.80, bottom=0.15, top=0.96)  # 进一步增加右侧留白，避免颜色条覆盖右轴刻度
    plt.tight_layout()  # 调整布局
    plt.show()
    plt.close(fig_dep)  # 关闭图形，释放内存

# %%
from statsmodels.nonparametric.smoothers_lowess import lowess
def bootstrap_lowess_ci(x, y, n_boot=200, frac=0.5, ci_level=0.95): # 定义一个函数，用bootstrap方法计算lowess平滑的置信区间
    """
    使用bootstrap方法计算lowess平滑的置信区间。
    参数说明:
    x (pd.Series): 模型的输入特征（自变量）。
    y (pd.Series): 模型的输出或真实值（因变量）。
    n_boot (int): bootstrap抽样的次数。次数越多，置信区间的估计越稳定，但计算成本也越高。默认为200次。
    frac (float): lowess平滑器中使用的样本比例。这个值控制平滑的程度，介于0和1之间。
                  值越小，曲线越贴近数据点；值越大，曲线越平滑。默认为0.5。
    ci_level (float): 置信区间的水平。例如，0.95表示计算95%的置信区间。默认为0.95。
    返回:
    tuple: (主平滑曲线, (x轴范围, 置信下界, 置信上界)) 或 (None, None)
    """
    if len(x) < 10: return None, None # 如果样本点太少，则不进行计算
    boot_lines = [] # 初始化一个列表，用于存储每次bootstrap抽样得到的平滑曲线
    x_range = np.linspace(x.min(), x.max(), 100) # 在x的范围内生成100个等间距点，用于插值
    for _ in range(n_boot): # 循环进行n_boot次bootstrap抽样
        sample_indices = np.random.choice(len(x), len(x), replace=True) # 有放回地抽取样本索引
        x_sample, y_sample = x[sample_indices], y[sample_indices] # 根据索引获取抽样数据
        sorted_indices = np.argsort(x_sample) # 对抽样的x值进行排序
        x_sorted, y_sorted = x_sample[sorted_indices], y_sample[sorted_indices] # 获取排序后的x和y
        if len(np.unique(x_sorted)) < 2: continue # 如果抽样后x的唯一值少于2个，则跳过此次循环
        smoothed = lowess(y_sorted, x_sorted, frac=frac) # 对抽样数据进行lowess平滑
        interp_func = np.interp(x_range, smoothed[:, 0], smoothed[:, 1]) # 将平滑结果插值到x_range上
        boot_lines.append(interp_func) # 将插值后的曲线添加到列表中
    if not boot_lines: return None, None # 如果未能生成任何bootstrap曲线，则返回None
    sorted_indices_orig = np.argsort(x) # 对原始x数据进行排序
    x_sorted_orig, y_sorted_orig = x[sorted_indices_orig], y[sorted_indices_orig] # 获取排序后的原始x和y
    main_smoothed = lowess(y_sorted_orig, x_sorted_orig, frac=frac) # 对完整的原始数据进行lowess平滑，作为主曲线
    boot_lines_arr = np.array(boot_lines) # 将bootstrap曲线列表转换为numpy数组
    alpha = (1 - ci_level) / 2 # 计算置信水平对应的alpha值
    lower_bound, upper_bound = np.quantile(boot_lines_arr, alpha, axis=0), np.quantile(boot_lines_arr, 1 - alpha,axis=0) # 计算每个点的上下置信边界
    return main_smoothed, (x_range, lower_bound, upper_bound) # 返回主平滑曲线和置信区间数据

# %%
def find_and_plot_crossings(ax, x_curve, y_curve, color, base_y_offset=0.9): # 定义一个函数，用于寻找并绘制曲线与y=0的交点（阈值）
    """
    在给定的Matplotlib Axes上寻找并绘制一条曲线与y=0轴的交点（阈值）。
    该函数通过线性插值精确计算交点位置，并用垂直虚线和文本标签在图上标记出来。
    文本标签会自动上下交错排列以避免重叠。
    参数说明:
    ax (matplotlib.axes.Axes): 要在其上绘图的Matplotlib子图对象。
    x_curve (np.array): 曲线的x坐标数组。
    y_curve (np.array): 曲线的y坐标数组。函数将寻找此曲线与y=0的交点。
    color (str): 用于绘制垂直线和文本背景的颜色。应与对应曲线的颜色匹配。
    base_y_offset (float): 控制文本标签垂直位置的基准偏移量，相对于y轴的高度。
                           默认为0.9，即从顶部向下10%的位置开始，然后交替向下排列。
    """
    sign_changes = np.where(np.diff(np.sign(y_curve)))[0] # 找到y值符号发生变化的位置
    for i, k in enumerate(sign_changes): # 遍历所有符号变化点
        x1, y1, x2, y2 = x_curve[k], y_curve[k], x_curve[k + 1], y_curve[k + 1] # 获取变化点前后的坐标
        if (y2 - y1) == 0: continue # 避免除以零
        x_root = x1 - y1 * (x2 - x1) / (y2 - y1) # 使用线性插值计算交点的x坐标（根）
        ax.axvline(x=x_root, color=color, linestyle='--', linewidth=1.8) # 在交点位置绘制一条垂直虚线（加粗）
        # y_text_position = ax.get_ylim()[1] * (base_y_offset - (i % 2) * 0.1) # 计算文本标签的y坐标，使其上下交错防止重叠
        ax.text(x_root, 0, f' {x_root:.2f} ', color='red', ha='center',
                va='center', fontsize=17, fontname='Times New Roman', fontweight='bold') # 在垂直线上方添加文本标签显示交点值

# %%
max_feature_plots = 5  # 仅绘制重要性Top5特征的单独SHAP图
plotted_count = 0       # 已成功绘制的特征数量计数器

for feature_name in top_feature_names_for_dependence: # 遍历重要性Top特征
    x_data_loop = X_train_[feature_name] # 获取当前特征的数值
    if not np.isfinite(x_data_loop).all(): # 检查特征值中是否包含NaN或无穷大等非有限值
        print(f"  -> 跳过特征: '{feature_name}'，因为它包含非有限值 (例如 NaN)。") # 如果有，则跳过该特征
        continue # 继续下一个循环
    i = int(columns.index(feature_name))
    y_data_shap_loop = shap_values_2d[:, i] # 获取当前特征对应的SHAP值
    print(y_data_shap_loop.shape)
    print(x_data_loop.shape)
    plot_shap_dependence(feature_name, x_data_loop.ravel(), y_data_shap_loop.ravel())
    plotted_count += 1.
    if plotted_count >= max_feature_plots: # 已绘制满10个特征后立即终止循环
        break

# %%
X_train_.shape

# %%
# 设置中文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 为类别1的每个特征创建决策图（这里只绘制重要性Top5个特征）
if False:
    for feature in top_feature_names_for_dependence:  # 遍历前5个特征
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(
            feature,
            shap_values_2d,
            X_train_,
            interaction_index=None,
            show=False
        )
        plt.title(f'{feature}特征决策图')
        plt.tight_layout()
        plt.show()

# %%
# 绘制单个样本的SHAP解释（Force Plot）
# 随机选择一个样本索引进行解释（覆盖全部有效样本范围）
sample_index = 10

import os
import json

# 为避免“matplotlib=True时不支持多样本”的报错，这里显式整理为单样本一维SHAP向量
shap_array = np.array(shap_values_2d)

# 运行时随机抽取样本：每次运行都可能不同
#if int(shap_array.shape[0]) <= 0:
#    raise ValueError(f"shap_values_2d 为空，无法随机选择样本: shape={tuple(shap_array.shape)}")
#_rng = np.random.default_rng()
#sample_index = int(_rng.integers(0, int(shap_array.shape[0])))
#print(f"[SHAP] 随机选择 sample_index={sample_index} (有效范围 0..{int(shap_array.shape[0]) - 1})")

if sample_index >= shap_array.shape[0]:
#    sample_index = int(shap_array.shape[0]) - 1
#if sample_index < 0:
    sample_index = 0

_sample_index_env = os.getenv("SHAP_SAMPLE_INDEX", "").strip()
if _sample_index_env:
    try:
        sample_index = int(_sample_index_env)
    except Exception:
        pass

_orig_row_env = os.getenv("SHAP_ORIG_ROW_INDEX", "").strip()
if _orig_row_env:
    try:
        _orig_row = int(_orig_row_env)
        _train_idx = globals().get("train_idx", None)
        if _train_idx is not None:
            _train_idx_arr = np.asarray(_train_idx).reshape(-1)
            _hits = np.where(_train_idx_arr == _orig_row)[0]
            if int(_hits.size) > 0:
                sample_index = int(_hits[0])
    except Exception:
        pass

if sample_index < 0:
    sample_index = 0
if sample_index >= int(shap_array.shape[0]):
    sample_index = int(shap_array.shape[0]) - 1

shap_sample = shap_array[sample_index]
shap_sample = shap_sample.reshape(-1)

expected_value = explainer.expected_value
MODEL_TARGET_INDEX = int(globals().get("MODEL_TARGET_INDEX", 0))
if isinstance(expected_value, (list, tuple, np.ndarray)):
    expected_value = np.array(expected_value).reshape(-1)
    if int(expected_value.size) > 0:
        t = int(MODEL_TARGET_INDEX)
        if t < 0:
            t = 0
        if t >= int(expected_value.size):
            t = int(expected_value.size) - 1
        expected_value = float(expected_value[t])
    else:
        expected_value = 0.0
else:
    expected_value = float(expected_value)

fx_force = float(expected_value + np.sum(shap_sample))

_model_out_scaled = None
_model_out_raw = None
try:
    _model_out_scaled = best_model(tensor_X_train[int(sample_index):int(sample_index) + 1, :]).detach().cpu().numpy()
    _model_out_scaled = np.asarray(_model_out_scaled, dtype=float).reshape(1, -1)
    _model_out_raw = scaler_y.inverse_transform(_model_out_scaled) if globals().get("scaler_y", None) is not None else _model_out_scaled
    _fx_direct_scaled = float(_model_out_scaled[0, int(MODEL_TARGET_INDEX)])
    _delta = float(_fx_direct_scaled) - float(fx_force)
    expected_value = float(expected_value) + float(_delta)
    fx_force = float(_fx_direct_scaled)
except Exception:
    _model_out_scaled = None
    _model_out_raw = None

def _get_y_affine(_scaler_y, target_index=0):
    if _scaler_y is None:
        return 0.0, 1.0
    mean_ = getattr(_scaler_y, "mean_", None)
    scale_ = getattr(_scaler_y, "scale_", None)
    if mean_ is None or scale_ is None:
        return 0.0, 1.0
    mean_ = np.asarray(mean_).reshape(-1)
    scale_ = np.asarray(scale_).reshape(-1)
    n_out = int(min(mean_.size, scale_.size))
    if n_out <= 0:
        return 0.0, 1.0
    t = int(target_index)
    if t < 0:
        t = 0
    if t >= n_out:
        t = n_out - 1
    y_mean = float(mean_[t])
    y_scale = float(scale_[t])
    if not np.isfinite(y_scale) or float(y_scale) == 0.0:
        y_scale = 1.0
    if not np.isfinite(y_mean):
        y_mean = 0.0
    return y_mean, y_scale


y_mean_t, y_scale_t = _get_y_affine(globals().get("scaler_y", None), target_index=MODEL_TARGET_INDEX)
expected_value_raw = float(expected_value * y_scale_t + y_mean_t)
shap_sample_raw = (shap_sample.astype(float) * float(y_scale_t)).reshape(-1)
fx_force_raw = float(expected_value_raw + float(np.sum(shap_sample_raw)))

_train_idx = globals().get("train_idx", None)
_orig_row_index = None
try:
    if _train_idx is not None:
        _train_idx_arr = np.asarray(_train_idx).reshape(-1)
        if 0 <= int(sample_index) < int(_train_idx_arr.size):
            _orig_row_index = int(_train_idx_arr[int(sample_index)])
except Exception:
    _orig_row_index = None

try:
    _ytr = globals().get("y_train_raw", None)
    if _ytr is not None:
        _ytr_arr = np.asarray(_ytr)
        if _ytr_arr.ndim == 1:
            _y_true_row = np.asarray([float(_ytr_arr[int(sample_index)])], dtype=float)
        else:
            _y_true_row = _ytr_arr[int(sample_index), :].astype(float)
    else:
        _y_true_row = None
except Exception:
    _y_true_row = None

_y_true_eval_raw = None
try:
    _yt_scaled = globals().get("y_train", None)
    if _yt_scaled is not None and globals().get("scaler_y", None) is not None:
        _yt_scaled_arr = np.asarray(_yt_scaled)
        if _yt_scaled_arr.ndim == 1:
            _yt_scaled_row = np.asarray([[float(_yt_scaled_arr[int(sample_index)])]], dtype=float)
        else:
            _yt_scaled_row = np.asarray(_yt_scaled_arr[int(sample_index):int(sample_index) + 1, :], dtype=float)
        _y_true_eval_raw = np.asarray(scaler_y.inverse_transform(_yt_scaled_row), dtype=float).reshape(-1)
except Exception:
    _y_true_eval_raw = None

_y_true_from_data = None
try:
    _y_all = globals().get("y", None)
    if _y_all is not None and _orig_row_index is not None:
        _y_all_arr = np.asarray(_y_all)
        if _y_all_arr.ndim == 1:
            _y_true_from_data = np.asarray([float(_y_all_arr[int(_orig_row_index)])], dtype=float)
        else:
            _y_true_from_data = np.asarray(_y_all_arr[int(_orig_row_index), :], dtype=float).reshape(-1)
except Exception:
    _y_true_from_data = None

try:
    if _model_out_scaled is not None and _model_out_raw is not None:
        _fx_direct_scaled = float(_model_out_scaled[0, int(MODEL_TARGET_INDEX)])
        _fx_direct_raw = float(_model_out_raw[0, int(MODEL_TARGET_INDEX)])
        if not np.isclose(float(fx_force), float(_fx_direct_scaled), rtol=1e-4, atol=1e-6):
            print(f"[SHAP][WARN] base+sum(shap) 与模型直出不一致: fx_scaled={float(fx_force):.6f} vs model_scaled={float(_fx_direct_scaled):.6f}")
        if not np.isclose(float(fx_force_raw), float(_fx_direct_raw), rtol=1e-4, atol=1e-6):
            print(f"[SHAP][WARN] raw 空间 base+sum(shap) 与模型直出不一致: fx_raw={float(fx_force_raw):.6f} vs model_raw={float(_fx_direct_raw):.6f}")
except Exception:
    pass

try:
    _y_true_str = "None" if _y_true_row is None else np.array2string(np.asarray(_y_true_row, dtype=float), precision=6, separator=",")
    _y_true_eval_str = "None" if _y_true_eval_raw is None else np.array2string(np.asarray(_y_true_eval_raw, dtype=float), precision=6, separator=",")
    _y_true_data_str = "None" if _y_true_from_data is None else np.array2string(np.asarray(_y_true_from_data, dtype=float), precision=6, separator=",")
    _pred_direct_raw_str = "None"
    _abs_err_str = "None"
    if _model_out_raw is not None:
        _pred_direct_raw = float(np.asarray(_model_out_raw, dtype=float).reshape(1, -1)[0, int(MODEL_TARGET_INDEX)])
        _pred_direct_raw_str = f"{_pred_direct_raw:.6f}"
        if _y_true_eval_raw is not None and int(np.asarray(_y_true_eval_raw).size) > int(MODEL_TARGET_INDEX):
            _abs_err = float(abs(_pred_direct_raw - float(np.asarray(_y_true_eval_raw, dtype=float).reshape(-1)[int(MODEL_TARGET_INDEX)])))
            _abs_err_str = f"{_abs_err:.6f}"
    print(
        f"[SHAP] sample_index_train={int(sample_index)} orig_row_index={_orig_row_index} "
        f"raw_target_col={int(globals().get('RAW_TARGET_COL', 0))} model_target_index={int(MODEL_TARGET_INDEX)} fx_raw={float(fx_force_raw):.6f} "
        f"y_true_trainraw={_y_true_str} y_true_evalraw={_y_true_eval_str} y_true_data={_y_true_data_str} "
        f"pred_model_raw={_pred_direct_raw_str} abs_err={_abs_err_str}"
    )
except Exception:
    pass

def _inverse_y_scalar(_scaler_y, y_scaled_scalar, target_index=0):
    if _scaler_y is None:
        return float(y_scaled_scalar)
    n_out = getattr(_scaler_y, "n_features_in_", None)
    if n_out is None:
        mean_ = getattr(_scaler_y, "mean_", None)
        n_out = int(mean_.shape[0]) if hasattr(mean_, "shape") else 1
    n_out = int(n_out) if int(n_out) > 0 else 1
    t = int(target_index)
    if t < 0:
        t = 0
    if t >= n_out:
        t = n_out - 1
    vec = np.zeros((1, n_out), dtype=float)
    vec[0, t] = float(y_scaled_scalar)
    inv = _scaler_y.inverse_transform(vec)
    return float(inv[0, t])


# 数值校验检查点：确认 scaled->raw 的逆变换公式正确
_expected_value_raw_check = _inverse_y_scalar(globals().get("scaler_y", None), expected_value, target_index=MODEL_TARGET_INDEX)
_fx_force_raw_check = _inverse_y_scalar(globals().get("scaler_y", None), fx_force, target_index=MODEL_TARGET_INDEX)
if not np.isfinite(_expected_value_raw_check) or not np.isfinite(_fx_force_raw_check):
    raise ValueError(f"scaled->raw 逆变换得到非有限值: expected={_expected_value_raw_check}, fx={_fx_force_raw_check}")
if not np.isclose(float(expected_value_raw), float(_expected_value_raw_check), rtol=1e-6, atol=1e-8):
    raise ValueError(
        f"expected_value raw 变换不一致: affine={float(expected_value_raw)} vs inverse_transform={float(_expected_value_raw_check)}"
    )
if not np.isclose(float(fx_force_raw), float(_fx_force_raw_check), rtol=1e-6, atol=1e-8):
    raise ValueError(
        f"f(x) raw 变换不一致: affine={float(fx_force_raw)} vs inverse_transform={float(_fx_force_raw_check)}"
    )
if not np.isclose(float(fx_force_raw), float(fx_force * float(y_scale_t) + float(y_mean_t)), rtol=1e-6, atol=1e-8):
    raise ValueError("f(x) raw 与 affine 公式不一致")
if not np.isclose(float(fx_force_raw), float(expected_value_raw + np.sum(shap_sample_raw)), rtol=1e-6, atol=1e-8):
    raise ValueError("f(x) raw 与 base+sum(shap) 不一致")

# 数值范围检查点：与训练集浓度范围对齐（仅告警，不中断）
_y_train_raw = globals().get("y_train_raw", None)
try:
    if _y_train_raw is not None:
        _y_arr = np.asarray(_y_train_raw)
        if _y_arr.ndim == 1:
            _y_col = _y_arr.reshape(-1)
        else:
            _y_col = _y_arr[:, int(MODEL_TARGET_INDEX)].reshape(-1)
        _p01, _p99 = np.nanpercentile(_y_col, [1, 99])
        if np.isfinite(_p01) and np.isfinite(_p99):
            if float(fx_force_raw) < float(_p01) or float(fx_force_raw) > float(_p99):
                print(
                    f"[SHAP][WARN] f(x)_raw={float(fx_force_raw):.6f} 超出训练集 1~99 分位范围 [{float(_p01):.6f}, {float(_p99):.6f}]，请结合业务判断是否合理"
                )
except Exception as _e:
    print("[SHAP][WARN] 浓度范围检查失败:", repr(_e))


def _inverse_x_row(_x_scaled_row):
    x_row = np.asarray(_x_scaled_row, dtype=float).reshape(1, -1)
    if bool(globals().get("USE_SPLIT_SCALING", False)):
        _p_idx = int(globals().get("PRESSURE_INDEX", x_row.shape[1] - 1))
        if _p_idx < 0:
            p_idx = int(x_row.shape[1] + _p_idx)
        else:
            p_idx = int(_p_idx)
        p_idx = max(0, min(p_idx, x_row.shape[1] - 1))
        spectra = x_row[:, :p_idx]
        pressure = x_row[:, p_idx:p_idx + 1]
        sx = globals().get("scaler_X_spectra", None)
        sp = globals().get("scaler_X_pressure", None)
        spectra_raw = sx.inverse_transform(spectra) if (sx is not None and int(spectra.shape[1]) > 0) else spectra
        pressure_raw = sp.inverse_transform(pressure) if sp is not None else pressure
        return np.hstack([spectra_raw, pressure_raw]).reshape(-1)

    sx = globals().get("scaler_X", None)
    if sx is None:
        return x_row.reshape(-1)
    return sx.inverse_transform(x_row).reshape(-1)


export_dir = os.path.join(os.getcwd(), "shap_exports")
os.makedirs(export_dir, exist_ok=True)

try:
    x_scaled_row = X_train_.iloc[int(sample_index)].to_numpy(dtype=float)
    x_raw_row = _inverse_x_row(x_scaled_row)
    df_inst = pd.DataFrame(
        {
            "feature": list(columns),
            "x_scaled": x_scaled_row.reshape(-1),
            "x_raw": x_raw_row.reshape(-1),
            "shap_value_scaled": shap_sample.reshape(-1),
            "shap_value_raw": shap_sample_raw.reshape(-1),
        }
    )
    df_inst.to_csv(
        os.path.join(export_dir, f"shap_instance_sample{int(sample_index)}.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    meta = {
        "sample_index": int(sample_index),
        "raw_target_col": int(globals().get("RAW_TARGET_COL", 0)),
        "model_target_index": int(MODEL_TARGET_INDEX),
        "target_index": int(MODEL_TARGET_INDEX),
        "y_mean": float(y_mean_t),
        "y_scale": float(y_scale_t),
        "expected_value_scaled": float(expected_value),
        "fx_scaled": float(fx_force),
        "expected_value_raw": float(expected_value_raw),
        "fx_raw": float(fx_force_raw),
    }
    with open(
        os.path.join(export_dir, f"shap_instance_sample{int(sample_index)}_meta.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
except Exception as _e:
    print("[WARN] 导出单样本SHAP数据失败:", repr(_e))

x_scaled_row_for_plot = None
try:
    x_scaled_row_for_plot = X_train_.iloc[int(sample_index)].to_numpy(dtype=float).reshape(-1)
except Exception:
    x_scaled_row_for_plot = None

shap_instance_explanation_raw = shap.Explanation(
    values=np.asarray(shap_sample_raw, dtype=float).reshape(-1),
    base_values=float(expected_value_raw),
    data=(x_scaled_row_for_plot if x_scaled_row_for_plot is not None else np.asarray(X_train_.iloc[int(sample_index)].to_numpy(dtype=float)).reshape(-1)),
    feature_names=columns,
)

force_fig = None
try:
    force_fig = shap.plots.force(shap_instance_explanation_raw, matplotlib=True, show=False)
except Exception:
    try:
        force_fig = shap.force_plot(
            float(expected_value_raw),
            np.asarray(shap_sample_raw, dtype=float).reshape(-1),
            (x_scaled_row_for_plot if x_scaled_row_for_plot is not None else X_train_.iloc[int(sample_index)].to_numpy(dtype=float).reshape(-1)),
            matplotlib=True,
            show=False,
        )
    except Exception:
        force_fig = None

_force_title = f"SHAP Force Plot | sample_index={int(sample_index)} | f(x)_raw={fx_force_raw:.6f} | f(x)_scaled={fx_force:.6f}"
try:
    if force_fig is not None and hasattr(force_fig, "axes") and force_fig.axes:
        force_fig.axes[0].set_title(_force_title)
except Exception:
    pass

try:
    if SHAP_SAVE_FIGURES:
        if force_fig is not None and hasattr(force_fig, "savefig"):
            force_fig.savefig("shap_force_instance.png", dpi=300, bbox_inches="tight")
        else:
            plt.gcf().savefig("shap_force_instance.png", dpi=300, bbox_inches="tight")
except Exception as _e:
    print("[WARN] 保存 shap_force_instance.png 失败:", repr(_e))

try:
    if force_fig is not None:
        plt.close(force_fig)
except Exception:
    pass

# %%
# 创建 shap.Explanation 对象（确保 SHAP 值为二维矩阵，避免 hclust 维度错误）
shap_matrix = np.array(shap_values_2d)

n_samples = shap_matrix.shape[0]
max_samples = min(500, n_samples)
shap_matrix_2d = shap_matrix[:max_samples]

if shap_matrix_2d.ndim > 2:
    shap_matrix_2d = np.squeeze(shap_matrix_2d)
if shap_matrix_2d.ndim == 1:
    shap_matrix_2d = shap_matrix_2d.reshape(1, -1)

base_value_for_explanation = explainer.expected_value
if isinstance(base_value_for_explanation, (list, tuple, np.ndarray)):
    base_value_for_explanation = np.asarray(base_value_for_explanation).reshape(-1)
    base_value_for_explanation = float(base_value_for_explanation[0]) if base_value_for_explanation.size else 0.0
else:
    base_value_for_explanation = float(base_value_for_explanation)

shap_explanation = shap.Explanation(
    values=shap_matrix_2d,
    base_values=base_value_for_explanation,
    data=X_train_.iloc[:max_samples, :],
    feature_names=columns
)

shap_matrix_2d_raw = np.asarray(shap_matrix_2d, dtype=float) * float(y_scale_t)
base_value_for_explanation_raw = float(base_value_for_explanation * float(y_scale_t) + float(y_mean_t))
shap_explanation_raw = shap.Explanation(
    values=shap_matrix_2d_raw,
    base_values=base_value_for_explanation_raw,
    data=X_train_.iloc[:max_samples, :],
    feature_names=columns,
)

try:
    np.savez_compressed(
        os.path.join(export_dir, f"shap_matrix_top{int(max_samples)}.npz"),
        shap_values_scaled=np.asarray(shap_matrix_2d, dtype=float),
        shap_values_raw=np.asarray(shap_matrix_2d_raw, dtype=float),
        feature_values=X_train_.iloc[:max_samples, :].to_numpy(dtype=float),
        feature_names=np.asarray(list(columns), dtype=object),
        base_value_scaled=np.asarray([float(base_value_for_explanation)], dtype=float),
        base_value_raw=np.asarray([float(base_value_for_explanation_raw)], dtype=float),
        y_mean=np.asarray([float(y_mean_t)], dtype=float),
        y_scale=np.asarray([float(y_scale_t)], dtype=float),
        raw_target_col=np.asarray([int(globals().get("RAW_TARGET_COL", 0))], dtype=int),
        model_target_index=np.asarray([int(MODEL_TARGET_INDEX)], dtype=int),
        target_index=np.asarray([int(MODEL_TARGET_INDEX)], dtype=int),
    )
except Exception as _e:
    print("[WARN] 导出SHAP矩阵数据失败:", repr(_e))

IMG_SCALE = 0.75
if IMG_SCALE < 0.6:
    IMG_SCALE = 0.6
if IMG_SCALE > 1.0:
    IMG_SCALE = 1.0


def _fit_fig_for_full_view(fig, left=0.28, right=0.98, bottom=0.12, top=0.92):
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=0.08)


plt.figure(figsize=(10 * IMG_SCALE, 5 * IMG_SCALE), dpi=300)
shap.plots.heatmap(
    shap_explanation_raw,
    cmap='inferno',
    show=False,
    max_display=10,
)
_fit_fig_for_full_view(plt.gcf(), left=0.10, right=0.96, bottom=0.12, top=0.94)
if SHAP_SAVE_FIGURES:
    plt.savefig("shap_heatmap_top10.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# 设置绘图样式
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']  # 同时支持英文和中文
plt.rcParams['axes.unicode_minus'] = False

sample_index_for_exp = sample_index
sample_index_for_exp = int(sample_index_for_exp)
exp_len = int(np.asarray(shap_explanation_raw.values).shape[0])
if exp_len <= 0:
    raise ValueError(f"shap_explanation 为空: values.shape={tuple(np.asarray(shap_explanation.values).shape)}")
if sample_index_for_exp >= exp_len:
    print(f"Warning: sample_index_for_exp={sample_index_for_exp} 越界(有效范围 0..{exp_len-1})，将使用最后一个索引")
    sample_index_for_exp = exp_len - 1
if sample_index_for_exp < 0:
    print(f"Warning: sample_index_for_exp={sample_index_for_exp} 小于0，将使用0")
    sample_index_for_exp = 0

_exp_row = shap_explanation_raw[sample_index_for_exp]
_base_val = np.asarray(_exp_row.base_values).reshape(-1)
_base_val = float(_base_val[0]) if _base_val.size else 0.0
_sv_sum = float(np.asarray(_exp_row.values).sum())
fx_local = float(_base_val + _sv_sum)

plt.figure(figsize=(10 * IMG_SCALE, 5 * IMG_SCALE), dpi=300)
shap.plots.bar(
    shap_instance_explanation_raw,
    show_data=True,
    show=False,
    max_display=12,
)
plt.title(f"SHAP Bar | sample_index={int(sample_index)} | f(x)_raw={float(fx_force_raw):.6f}")
_fit_fig_for_full_view(plt.gcf(), left=0.34, right=0.98, bottom=0.16, top=0.90)
if SHAP_SAVE_FIGURES:
    plt.savefig("shap_bar_instance.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
shap_instance_explanation_raw_no_data = shap.Explanation(
    values=np.asarray(shap_sample_raw, dtype=float).reshape(-1),
    base_values=float(expected_value_raw),
    data=None,
    feature_names=columns,
)

plt.figure(figsize=(10 * IMG_SCALE, 5 * IMG_SCALE), dpi=300)
shap.plots.waterfall(
    shap_instance_explanation_raw_no_data,
    show=False,
    max_display=12,
)
_ax_wf = plt.gca()
_x0, _x1 = _ax_wf.get_xlim()
_dx = float(_x1 - _x0)
_min_x = float(_x0 + 0.25 * _dx)
_shift = float(0.12 * _dx)
for _t in list(getattr(_ax_wf, "texts", [])):
    try:
        _c = str(_t.get_color()).lower()
        _pos = _t.get_position()
        _tx = float(_pos[0])
        if _c in {"blue", "tab:blue", "#1f77b4"} and _tx <= _min_x:
            _t.set_x(_tx + _shift)
    except Exception:
        pass
plt.title(f"SHAP Waterfall | f(x)_raw={float(fx_force_raw):.6f}")

# 第二张图统一字体：Times New Roman + Bold
_wf_font = FontProperties(family="Times New Roman", weight="bold")
_ax_wf.title.set_fontproperties(_wf_font)
_ax_wf.xaxis.label.set_fontproperties(_wf_font)
_ax_wf.yaxis.label.set_fontproperties(_wf_font)
for _tick in _ax_wf.get_xticklabels() + _ax_wf.get_yticklabels():
    _tick.set_fontproperties(_wf_font)
for _txt in list(getattr(_ax_wf, "texts", [])):
    _txt.set_fontproperties(_wf_font)

_fit_fig_for_full_view(plt.gcf(), left=0.34, right=0.98, bottom=0.16, top=0.90)
if SHAP_SAVE_FIGURES:
    plt.savefig("shap_waterfall_instance.png", dpi=300, bbox_inches="tight")
plt.show()

topk_heatmap = 8
manual_heatmap_features = [d[0] for d in feature_importance_data[:topk_heatmap]]
manual_heatmap_idx = [columns.index(name) for name in manual_heatmap_features]
manual_n_samples = min(200, shap_matrix_2d_raw.shape[0])
manual_shap_sub = shap_matrix_2d_raw[:manual_n_samples, :][:, manual_heatmap_idx].T
manual_vlim = np.nanmax(np.abs(manual_shap_sub)) if np.isfinite(np.nanmax(np.abs(manual_shap_sub))) else 1.0

plt.figure(figsize=(10 * IMG_SCALE, 4 * IMG_SCALE), dpi=300)
im_manual = plt.imshow(
    manual_shap_sub,
    aspect="auto",
    cmap="seismic",
    vmin=-manual_vlim,
    vmax=manual_vlim,
)
plt.yticks(range(topk_heatmap), manual_heatmap_features)
plt.xlabel("Instance index")
plt.colorbar(im_manual, label="SHAP value")
plt.title(f"Top{int(topk_heatmap)} features SHAP heatmap")
plt.tight_layout()
if SHAP_SAVE_FIGURES:
    plt.savefig("shap_heatmap_top5_manual.png", dpi=300, bbox_inches="tight")
plt.show()


import seaborn as sns
subset_cols = columns[:8]
sub_df = X_train_[subset_cols]
sns.set_theme(style="whitegrid")
n_sub = int(len(subset_cols))
ncols = 4
nrows = int(np.ceil(n_sub / float(ncols)))
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 4.0 * nrows), dpi=300)
axes2 = np.asarray(axes2).reshape(-1)
for i, col in enumerate(subset_cols):
    sns.boxplot(y=sub_df[col], ax=axes2[i], color="#1f77b4")
    axes2[i].set_title(f"前8个特征子集可视化 - {col}", fontsize=12)
for j in range(n_sub, int(axes2.size)):
    axes2[j].set_visible(False)
fig2.suptitle("前8个特征子集可视化（箱线图）", fontsize=16)
plt.tight_layout()




