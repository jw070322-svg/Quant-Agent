# train_a_stock.py
import os, warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_CN_DIR = os.path.join(PROJECT_ROOT, 'data', 'cn')
DATA_US_DIR = os.path.join(PROJECT_ROOT, 'data', 'us')
MODEL_CN_DIR = os.path.join(PROJECT_ROOT, 'models', 'cn')
MODEL_US_DIR = os.path.join(PROJECT_ROOT, 'models', 'us')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# ==================== 配置 ====================
TICKER = '603986.SH'          # 平安银行示例
DATA_DIR = r"F:\PythonProject\stock_data"
MODEL_SAVE_PATH = f'model_{TICKER}.pkl'
FEATURES_SAVE_PATH = f'features_{TICKER}.txt'

FEATURES = [
    'MA5', 'MA10', 'MA20',
    'MACD', 'MACD_hist',
    'K', 'D', 'J',
    'RSI',
    'BB_width', 'BB_position',
    'volatility_5', 'volatility_20',
    'close_vs_MA20', 'volume_ratio',
    'roc_7', 'roc_14'
]

# ==================== 数据加载 ====================
def load_stock_data(ticker):
    local_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"本地文件 {local_path} 不存在")
    df = pd.read_csv(local_path)
    if df.empty:
        raise ValueError(f"{ticker}.csv 为空")

    # 兼容 Tushare 列名
    rename_map = {
        'trade_date': '日期', 'open': '开盘', 'high': '最高',
        'low': '最低', 'close': '收盘', 'vol': '成交量'
    }
    df.rename(columns=rename_map, inplace=True)

    # 日期转换
    if df['日期'].dtype == 'int64' or df['日期'].dtype == 'float64':
        df['日期'] = pd.to_datetime(df['日期'].astype(str), format='%Y%m%d')
    else:
        df['日期'] = pd.to_datetime(df['日期'])

    required = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
    df = df[required].copy()
    df.sort_values('日期', inplace=True)
    df.reset_index(drop=True, inplace=True)
    # ===== 前复权处理 =====
    if 'adj_factor' in df.columns:
        last_factor = df['adj_factor'].iloc[-1]          # 最新复权因子
        for col in ['开盘', '最高', '最低', '收盘']:     # 使用标准化后的中文列名
            if col in df.columns:
                df[col] = df[col] * (df['adj_factor'] / last_factor)
        df.drop(columns=['adj_factor'], inplace=True)    # 删除不再需要的复权因子列
    # =====================
    return df

# ==================== 特征计算 (与之前完全一致) ====================
def compute_features(df):
    df = df.copy()
    df['MA5']  = df['收盘'].rolling(5).mean()
    df['MA10'] = df['收盘'].rolling(10).mean()
    df['MA20'] = df['收盘'].rolling(20).mean()
    exp12 = df['收盘'].ewm(span=12, adjust=False).mean()
    exp26 = df['收盘'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    low_14 = df['最低'].rolling(14).min()
    high_14 = df['最高'].rolling(14).max()
    rsv = (df['收盘'] - low_14) / (high_14 - low_14) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3*df['K'] - 2*df['D']
    delta = df['收盘'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1+rs))
    ma20 = df['收盘'].rolling(20).mean()
    bb_std = df['收盘'].rolling(20).std()
    df['BB_width'] = 2 * bb_std
    df['BB_position'] = (df['收盘'] - (ma20 - 2*bb_std)) / (4*bb_std + 1e-9)
    df['roc_7'] = df['收盘'].pct_change(7)
    df['roc_14'] = df['收盘'].pct_change(14)
    df['returns'] = df['收盘'].pct_change()
    df['log_return'] = np.log(df['收盘'] / df['收盘'].shift(1))
    df['volatility_5'] = df['returns'].rolling(5).std()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['close_vs_MA20'] = (df['收盘'] - df['MA20']) / df['MA20']
    df['volume_ratio'] = df['成交量'] / df['成交量'].rolling(20).mean()
    return df

# ==================== 构造标签 (未来 1 日涨幅 > 1%) ====================
def prepare_full_data(df):
    df = df.copy()
    # ⚠️ A股 T+1，预测下一天
    df['future_return'] = df['收盘'].shift(-1) / df['收盘'] - 1
    df['target'] = (df['future_return'] > 0.01).astype(int)
    df_clean = df.dropna(subset=FEATURES + ['target']).reset_index(drop=True)
    for col in FEATURES:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna(subset=FEATURES + ['target']).reset_index(drop=True)
    return df_clean

# ==================== 主流程 ====================
print(f"加载数据 {TICKER} ...")
data = load_stock_data(TICKER)
print(f"原始数据: {len(data)} 条, 日期 {data['日期'].min().date()} ~ {data['日期'].max().date()}")
data = compute_features(data)
df_clean = prepare_full_data(data)
print(f"清洗后样本: {len(df_clean)} 条")

# 按时间 8:2 分割
split = int(len(df_clean) * 0.8)
train_df = df_clean.iloc[:split]
test_df = df_clean.iloc[split:]
print(f"训练集: {len(train_df)} 条, 测试集: {len(test_df)} 条")

X_train = train_df[FEATURES].astype('float32')
y_train = train_df['target'].astype(int)
X_test = test_df[FEATURES].astype('float32')
y_test = test_df['target'].astype(int)

# 训练模型
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 评估
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {acc:.4f}")

# 简单回测：只用 prob > 0.5 做多
test = test_df.copy()
test['prob'] = y_prob
test['position'] = np.where(test['prob'] > 0.5, 1, 0)
test['ret'] = test['position'].shift(1) * test['log_return']
test['ret_net'] = test['ret'] - test['position'].diff().abs() * 0.0005  # A股千分之1.5手续费+印花税大致
cum = test['ret_net'].sum() * 100
sharpe = (test['ret_net'].mean() / test['ret_net'].std()) * np.sqrt(252) if test['ret_net'].std() != 0 else 0.0
max_dd = (test['ret_net'].cumsum().cummax() - test['ret_net'].cumsum()).max() * 100
print(f"简单回测(>0.5做多): 累计收益 {cum:.2f}%, 夏普 {sharpe:.2f}, 最大回撤 {max_dd:.2f}%")

# 保存模型和特征
joblib.dump(model, MODEL_SAVE_PATH)
with open(FEATURES_SAVE_PATH, 'w') as f:
    f.write(','.join(FEATURES))
print(f"\n✅ 模型已保存为 {MODEL_SAVE_PATH}")
print(f"✅ 特征列表已保存为 {FEATURES_SAVE_PATH}")

# 收益曲线（可选）
plt.figure(figsize=(10,4))
(1 + test['ret_net']).cumprod().plot(label='策略净值')
plt.title(f'{TICKER} 简单回测净值')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('a_stock_equity.png', dpi=150)
plt.show()