# robust_walk_forward_short.py  (日期修复 + 合理窗口)
import os, warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
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
TICKERS = ['AAPL', 'AMZN', 'GOOGL']
DATA_DIR = r"F:\PythonProject\stock_data"

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

THRESHOLD_RANGE = np.arange(0.45, 0.80, 0.05)

# 滚动窗口大小（交易日）
WINDOW_TRAIN = 750      # 约3年
WINDOW_VAL   = 125
WINDOW_TEST  = 250
STEP_SIZE    = 125

# ==================== 数据加载（修复日期） ====================
def load_stock_data(ticker):
    local_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"本地文件 {local_path} 不存在")
    df = pd.read_csv(local_path)
    if df.empty:
        raise ValueError(f"{ticker}.csv 为空")

    # 列名映射
    rename_map = {
        'trade_date': '日期', 'open': '开盘', 'high': '最高',
        'low': '最低', 'close': '收盘', 'vol': '成交量','adj_factor': 'adj_factor',
        'Date': '日期', 'Open': '开盘', 'High': '最高',
        'Low': '最低', 'Close': '收盘', 'Volume': '成交量',
        'date': '日期', 'volume': '成交量'
    }
    df.rename(columns=rename_map, inplace=True)

    # 强制日期转换：Tushare 整数 YYYYMMDD
    if df['日期'].dtype == 'int64' or df['日期'].dtype == 'float64':
        df['日期'] = pd.to_datetime(df['日期'].astype(str), format='%Y%m%d')
    else:
        df['日期'] = pd.to_datetime(df['日期'])

    required = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
    df = df[required].copy()
    df.sort_values('日期', inplace=True)
    df.reset_index(drop=True, inplace=True)
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
    # ===== 前复权处理 =====
    if 'adj_factor' in df.columns:
        last_factor = df['adj_factor'].iloc[-1]          # 最新复权因子
        for col in ['开盘', '最高', '最低', '收盘']:     # 使用标准化后的中文列名
            if col in df.columns:
                df[col] = df[col] * (df['adj_factor'] / last_factor)
        df.drop(columns=['adj_factor'], inplace=True)    # 删除不再需要的复权因子列
    # =====================
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

# ==================== 滚动窗口主循环 ====================
all_test_returns = []
performance_report = []

for ticker in TICKERS:
    print(f"\n{'='*50}\n处理 {ticker}...")
    data = load_stock_data(ticker)
    print(f"  原始日期: {data['日期'].min().date()} ~ {data['日期'].max().date()}, 行数: {len(data)}")
    data = compute_features(data)
    df_clean = prepare_full_data(data)
    print(f"  清洗后日期: {df_clean['日期'].min().date()} ~ {df_clean['日期'].max().date()}, 样本: {len(df_clean)}")

    n = len(df_clean)
    if n < WINDOW_TRAIN + WINDOW_VAL + WINDOW_TEST:
        print(f"  ⚠️ 数据不足以构成一个窗口，跳过")
        continue

    ticker_returns = []
    start_idx = 0
    while start_idx + WINDOW_TRAIN + WINDOW_VAL + WINDOW_TEST <= n:
        train_end = start_idx + WINDOW_TRAIN
        val_end = train_end + WINDOW_VAL
        test_end = val_end + WINDOW_TEST

        train_df = df_clean.iloc[start_idx:train_end]
        val_df   = df_clean.iloc[train_end:val_end]
        test_df  = df_clean.iloc[val_end:test_end]

        X_train = train_df[FEATURES].astype('float32')
        y_train = train_df['target'].astype(int)
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train)

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

        test_prob = model.predict_proba(test_df[FEATURES].astype('float32'))[:, 1]
        test_log_ret = test_df['log_return'].values
        test_dates = test_df['日期'].values
        bt_test = pd.DataFrame({'日期': test_dates, 'prob': test_prob, 'log_return': test_log_ret})
        bt_test['position'] = np.where(bt_test['prob'] > best_thr, 1, 0)
        bt_test['strategy_ret'] = bt_test['position'].shift(1) * bt_test['log_return']
        bt_test['turnover'] = bt_test['position'].diff().abs()
        bt_test['strategy_ret_net'] = bt_test['strategy_ret'] - bt_test['turnover'] * 0.00015

        ticker_returns.append(bt_test.set_index('日期')['strategy_ret_net'])
        start_idx += STEP_SIZE

    if ticker_returns:
        combined_ticker = pd.concat(ticker_returns).sort_index()
        daily_ret = combined_ticker.dropna()
        cum_ret = daily_ret.sum() * 100
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() != 0 else 0.0
        max_dd = (daily_ret.cumsum().cummax() - daily_ret.cumsum()).max() * 100
        win_rate = (daily_ret > 0).mean()

        performance_report.append({
            'Ticker': ticker,
            '累计收益(%)': round(cum_ret, 2),
            '夏普比率': round(sharpe, 2),
            '最大回撤(%)': round(max_dd, 2),
            '胜率': round(win_rate, 4)
        })
        combined_ticker.name = ticker
        all_test_returns.append(combined_ticker)
        print(f"  {ticker} 样本外累计收益: {cum_ret:.2f}%, 夏普: {sharpe:.2f}")
    else:
        print(f"  {ticker} 无足够窗口")

# ==================== 等权组合 ====================
if all_test_returns:
    combined = pd.concat(all_test_returns, axis=1).dropna()
    port_ret = combined.mean(axis=1)
    port_cum = port_ret.cumsum() * 100
    port_sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(252) if port_ret.std() != 0 else 0.0
    port_dd = (port_cum.cummax() - port_cum).max()
    port_win = (port_ret > 0).mean()

    print("\n===== 等权组合（短期聚焦样本外） =====")
    print(f"组合累计收益: {port_cum.iloc[-1]:.2f}%")
    print(f"组合夏普比率: {port_sharpe:.2f}")
    print(f"组合最大回撤: {port_dd:.2f}%")
    print(f"组合胜率: {port_win:.4f}")

print("\n" + "=" * 80)
print("                    📈 短期聚焦 Walk-Forward 样本外绩效报告")
print("=" * 80)
header = f"{'标的':<8} {'累计收益%':>10} {'夏普':>8} {'最大回撤%':>10} {'胜率':>8}"
print(header)
print("-" * 80)
for r in performance_report:
    print(f"{r['Ticker']:<8} {r['累计收益(%)']:>10.2f} {r['夏普比率']:>8.2f} {r['最大回撤(%)']:>10.2f} {r['胜率']:>8.4f}")
print("-" * 80)
if all_test_returns:
    print(f"  等权组合样本外累计收益: {port_cum.iloc[-1]:.2f}%")
    print(f"  等权组合样本外夏普: {port_sharpe:.2f}")
    print(f"  等权组合样本外最大回撤: {port_dd:.2f}%")
    print(f"  等权组合样本外胜率: {port_win:.4f}")
print("=" * 80)

if all_test_returns and not port_cum.empty:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(port_cum.index, port_cum, color='steelblue', linewidth=1.2)
    ax1.fill_between(port_cum.index, 0, port_cum, where=(port_cum > 0), color='steelblue', alpha=0.1)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.set_title('Short-Term Focused Walk-Forward (Long Only)')
    ax1.grid(True, alpha=0.3)

    dd = port_cum - port_cum.cummax()
    ax2.fill_between(port_cum.index, dd, 0, color='lightcoral', alpha=0.6)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('short_focus_oos.png', dpi=150)
    plt.show()

print("\n✅ 短期聚焦滚动验证完成")