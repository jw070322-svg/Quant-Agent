# live_strategy.py
import os, warnings, json
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
import joblib

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
TICKERS = ['AAPL', 'AMZN', 'GOOGL']
DATA_DIR = r"F:\PythonProject\stock_data"
MODEL_DIR = '.'  # 保存模型的目录

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

XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

# 训练与验证窗口大小
TRAIN_DAYS = 750
VAL_DAYS = 125

# 阈值搜索范围
THRESHOLD_RANGE = np.arange(0.45, 0.80, 0.05)

# ==================== 数据加载与特征工程（同上） ====================
def load_stock_data(ticker):
    local_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"本地文件 {local_path} 不存在")
    df = pd.read_csv(local_path)
    rename_map = {
        'trade_date': '日期', 'open': '开盘', 'high': '最高',
        'low': '最低', 'close': '收盘', 'vol': '成交量',
        'Date': '日期', 'Open': '开盘', 'High': '最高',
        'Low': '最低', 'Close': '收盘', 'Volume': '成交量',
        'date': '日期', 'volume': '成交量'
    }
    df.rename(columns=rename_map, inplace=True)
    # 强制转换日期
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

def prepare_full_data(df):
    df = df.copy()
    df['future_return'] = df['收盘'].shift(-2) / df['收盘'] - 1
    df['target'] = (df['future_return'] > 0.01).astype(int)
    df_clean = df.dropna(subset=FEATURES + ['target']).reset_index(drop=True)
    for col in FEATURES:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna(subset=FEATURES + ['target']).reset_index(drop=True)
    return df_clean

# ==================== 每只股票训练并生成信号 ====================
signals = {}
for ticker in TICKERS:
    print(f"训练 {ticker}...")
    data = load_stock_data(ticker)
    data = compute_features(data)
    df_clean = prepare_full_data(data)

    n = len(df_clean)
    if n < TRAIN_DAYS + VAL_DAYS:
        print(f"  {ticker} 数据不足，跳过")
        continue

    # 分割训练集和验证集
    train_start = n - TRAIN_DAYS - VAL_DAYS
    val_start = train_start + TRAIN_DAYS
    train_df = df_clean.iloc[train_start:val_start]
    val_df = df_clean.iloc[val_start:]

    # 训练模型
    X_train = train_df[FEATURES].astype('float32')
    y_train = train_df['target'].astype(int)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    # 在验证集上选阈值
    val_prob = model.predict_proba(val_df[FEATURES].astype('float32'))[:, 1]
    val_log_ret = val_df['log_return'].values
    best_cum = -np.inf
    best_thr = 0.5
    for thr in THRESHOLD_RANGE:
        bt_val = pd.DataFrame({'prob': val_prob, 'log_return': val_log_ret})
        bt_val['position'] = np.where(bt_val['prob'] > thr, 1, 0)
        bt_val['strategy_ret'] = bt_val['position'].shift(1) * bt_val['log_return']
        bt_val['turnover'] = bt_val['position'].diff().abs()
        bt_val['strategy_ret_net'] = bt_val['strategy_ret'] - bt_val['turnover'] * 0.00015
        cum = bt_val['strategy_ret_net'].sum() * 100
        if cum > best_cum:
            best_cum = cum
            best_thr = thr

    # 最新一日的特征与预测
    latest = df_clean.iloc[-1:]
    X_latest = latest[FEATURES].astype('float32')
    prob = model.predict_proba(X_latest)[0, 1]
    signal = 1 if prob > best_thr else 0

    signals[ticker] = {
        'date': latest['日期'].iloc[0].strftime('%Y-%m-%d'),
        'close': float(latest['收盘'].iloc[0]),
        'prob': round(float(prob), 4),
        'threshold': best_thr,
        'signal': signal,
        'action': '买入' if signal == 1 else '持有/空仓'
    }

    # 保存模型和阈值
    joblib.dump(model, os.path.join(MODEL_DIR, f'live_model_{ticker}.pkl'))
    with open(os.path.join(MODEL_DIR, f'live_threshold_{ticker}.txt'), 'w') as f:
        f.write(str(best_thr))

    print(f"  {ticker} 最新日期: {signals[ticker]['date']}, 收盘: {signals[ticker]['close']:.2f}, "
          f"上涨概率: {signals[ticker]['prob']:.2%}, 阈值: {best_thr:.2f}, 信号: {signals[ticker]['action']}")

# ==================== 汇总输出 ====================
print("\n" + "=" * 60)
print("                    📈 今日交易信号")
print("=" * 60)
for t, s in signals.items():
    print(f"{t}: 日期 {s['date']} 收盘 {s['close']:.2f}  上涨概率 {s['prob']:.2%}  "
          f"阈值 {s['threshold']:.2f}  建议: {s['action']}")
print("=" * 60)

# 可选：将信号保存为 JSON，供其他系统读取
with open('today_signals.json', 'w', encoding='utf-8') as f:
    json.dump(signals, f, ensure_ascii=False, indent=2)
print("\n✅ 信号已保存至 today_signals.json")