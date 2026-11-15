# scripts/training/individual_models.py
import os, warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==================== 项目根目录定位 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'us')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'us')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== 配置 ====================
TICKERS = ['AAPL', 'GOOGL', 'AMZN']   # 只保留已有数据的股票
START_DATE = '2019-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')

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

# GOOGL 的最优交易参数（来自之前的 tune_googl.py）
GOOGL_BEST_THR = 0.65
GOOGL_BEST_SCALE = 1.0

# ==================== 数据加载（高度兼容） ====================
def load_stock_data(ticker):
    local_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(local_path):
        print(f"  ⚠️ {ticker} 本地文件不存在，跳过")
        return None
    df = pd.read_csv(local_path)
    if df.empty:
        print(f"  ⚠️ {ticker}.csv 为空，跳过")
        return None

    rename_map = {
        'date': '日期', 'open': '开盘', 'high': '最高', 'low': '最低',
        'close': '收盘', 'volume': '成交量',
        'Date': '日期', 'Open': '开盘', 'High': '最高',
        'Low': '最低', 'Close': '收盘', 'Volume': '成交量',
        'trade_date': '日期', 'vol': '成交量'
    }
    df.rename(columns=rename_map, inplace=True)

    required = {'日期', '开盘', '最高', '最低', '收盘'}
    missing = required - set(df.columns)
    if missing:
        print(f"  ❌ {ticker} 缺少必要列: {missing}，现有列: {list(df.columns)}")
        return None

    if '成交量' not in df.columns:
        df['成交量'] = 0

    df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']].copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df.sort_values('日期', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  📂 {ticker} 加载成功，{len(df)} 行")
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

def prepare_data(df):
    df = df.copy()
    df['future_return'] = df['收盘'].shift(-2) / df['收盘'] - 1
    df['target'] = (df['future_return'] > 0.01).astype(int)
    df_clean = df.dropna(subset=FEATURES + ['target']).reset_index(drop=True)
    for col in FEATURES:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna(subset=FEATURES + ['target']).reset_index(drop=True)
    X = df_clean[FEATURES].astype('float32')
    y = df_clean['target'].astype(int)
    dates = df_clean['日期']
    log_return = df_clean['log_return']
    return X, y, dates, log_return

# ==================== 逐只训练与回测 ====================
results = []
portfolio_returns = []

for ticker in TICKERS:
    print(f"\n{'='*50}\n处理 {ticker}...")
    data = load_stock_data(ticker)
    if data is None:
        continue

    # 截断近4年
    cutoff = data['日期'].max() - pd.DateOffset(years=4)
    data = data[data['日期'] > cutoff]
    if len(data) < 100:
        print(f"  ⚠️ {ticker} 数据量不足，跳过")
        continue

    data = compute_features(data)
    X, y, dates, log_ret = prepare_data(data)
    if len(X) < 50:
        print(f"  ⚠️ {ticker} 有效样本不足，跳过")
        continue

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    test_dates = dates.iloc[split:].copy()
    test_log_ret = log_ret.iloc[split:].copy()

    # ---------- GOOGL 特殊处理：加载优化模型 ----------
    if ticker == 'GOOGL':
        model_path = os.path.join(MODEL_DIR, 'model_GOOGL.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"  加载已有优化模型: {model_path}")
        else:
            print(f"  优化模型不存在，重新训练...")
            model = xgb.XGBClassifier(**XGB_PARAMS)
            model.fit(X_train, y_train)
        # 使用预存的最优交易参数
        best_thr = GOOGL_BEST_THR
        best_s = GOOGL_BEST_SCALE
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, (y_prob > 0.5).astype(int))
        print(f"  测试集准确率: {acc:.4f}")
        print(f"  使用固定参数：阈值={best_thr:.2f}, 仓位系数={best_s:.2f}")
    else:
        # ---------- 其他股票：正常训练并搜索参数 ----------
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, (y_prob > 0.5).astype(int))
        print(f"  测试集准确率: {acc:.4f}")

        # 双向交易参数搜索
        THRESHOLD_RANGE = [0.50, 0.55, 0.60, 0.65]
        SCALE_RANGE = np.arange(1.5, 4.0, 0.5)
        bt = pd.DataFrame({
            '日期': test_dates.values,
            'prob': y_prob,
            'log_return': test_log_ret.values
        })
        best_cum = -np.inf
        best_thr = 0.5
        best_s = 2.0
        for thr in THRESHOLD_RANGE:
            for s in SCALE_RANGE:
                bt['position'] = 0
                bt.loc[bt['prob'] > thr, 'position'] = 1
                bt.loc[bt['prob'] < 1 - thr, 'position'] = -1
                bt['position'] = bt['position'] * s
                bt['position'] = np.clip(bt['position'], -1, 1)
                bt['strategy_ret'] = bt['position'].shift(1) * bt['log_return']
                bt['turnover'] = bt['position'].diff().abs()
                bt['strategy_ret_net'] = bt['strategy_ret'] - bt['turnover'] * 0.00015
                cum = bt['strategy_ret_net'].sum() * 100
                if cum > best_cum:
                    best_cum = cum
                    best_thr = thr
                    best_s = s
        print(f"  最优阈值: {best_thr:.2f}, 最优仓位系数: {best_s:.2f}")

    # ---------- 使用最终参数计算绩效 ----------
    bt = pd.DataFrame({
        '日期': test_dates.values,
        'prob': y_prob,
        'log_return': test_log_ret.values
    })
    bt['position'] = 0
    bt.loc[bt['prob'] > best_thr, 'position'] = 1
    bt.loc[bt['prob'] < 1 - best_thr, 'position'] = -1
    bt['position'] = bt['position'] * best_s
    bt['position'] = np.clip(bt['position'], -1, 1)
    bt['strategy_ret'] = bt['position'].shift(1) * bt['log_return']
    bt['turnover'] = bt['position'].diff().abs()
    bt['strategy_ret_net'] = bt['strategy_ret'] - bt['turnover'] * 0.00015

    cum = bt['strategy_ret_net'].sum() * 100
    daily_ret = bt['strategy_ret_net'].dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() != 0 else 0.0
    max_dd = (daily_ret.cumsum().cummax() - daily_ret.cumsum()).max() * 100
    traded = bt[bt['position'].shift(1) != 0]
    win_rate = (traded['strategy_ret_net'] > 0).mean() if len(traded) > 0 else 0.0

    results.append({
        'Ticker': ticker,
        '阈值': best_thr,
        '仓位系数': best_s,
        '累计收益(%)': round(cum, 2),
        '夏普比率': round(sharpe, 2),
        '最大回撤(%)': round(max_dd, 2),
        '胜率': round(win_rate, 4),
        '准确率': round(acc, 4)
    })

    # 保存模型（仅当不是加载已有模型时，才保存，防止覆盖优化模型）
    if ticker != 'GOOGL':
        joblib.dump(model, os.path.join(MODEL_DIR, f'model_{ticker}.pkl'))
        with open(os.path.join(MODEL_DIR, f'features_{ticker}.txt'), 'w') as f:
            f.write(','.join(FEATURES))

    portfolio_returns.append(bt[['日期', 'strategy_ret_net']].set_index('日期').rename(columns={'strategy_ret_net': ticker}))
    print(f"  {ticker}: 累计收益 {cum:.2f}%, 夏普 {sharpe:.2f}, 最大回撤 {max_dd:.2f}%")

# ==================== 等权组合回测 ====================
if portfolio_returns:
    combined = pd.concat(portfolio_returns, axis=1).dropna()
    if not combined.empty:
        port_ret = combined.mean(axis=1)
        port_cum = port_ret.cumsum() * 100
        port_sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(252) if port_ret.std() != 0 else 0.0
        port_dd = (port_cum.cummax() - port_cum).max()
        port_win = (port_ret > 0).mean()
    else:
        port_cum = pd.Series()
else:
    port_cum = pd.Series()

# ==================== 可视化报告 ====================
print("\n" + "=" * 90)
print("                          📈 独立建模策略绩效报告 (GOOGL已优化)")
print("=" * 90)
header = f"{'标的':<8} {'阈值':<6} {'仓位系数':<8} {'累计收益%':>10} {'夏普':>8} {'最大回撤%':>10} {'胜率':>8} {'准确率':>8}"
print(header)
print("-" * 90)
for r in results:
    print(f"{r['Ticker']:<8} {r['阈值']:<6.2f} {r['仓位系数']:<8.2f} {r['累计收益(%)']:>10.2f} {r['夏普比率']:>8.2f} {r['最大回撤(%)']:>10.2f} {r['胜率']:>8.4f} {r['准确率']:>8.4f}")
print("-" * 90)
if not port_cum.empty:
    print(f"  等权组合累计收益: {port_cum.iloc[-1]:.2f}%")
    print(f"  组合夏普比率: {port_sharpe:.2f}")
    print(f"  组合最大回撤: {port_dd:.2f}%")
    print(f"  组合胜率: {port_win:.4f}")
print("=" * 90)

# 保存组合收益图
if not port_cum.empty:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(port_cum.index, port_cum, color='steelblue', linewidth=1.2, label='等权组合')
    ax1.fill_between(port_cum.index, 0, port_cum, where=(port_cum > 0), color='steelblue', alpha=0.1)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_ylabel('累积收益 (%)')
    ax1.legend(loc='upper left')
    ax1.set_title('独立模型等权组合累积收益曲线 (含优化GOOGL)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    dd = (port_cum - port_cum.cummax())
    ax2.fill_between(port_cum.index, dd, 0, color='lightcoral', alpha=0.6)
    ax2.set_ylabel('回撤 (%)')
    ax2.set_xlabel('日期')
    ax2.set_title('组合回撤曲线', fontsize=12)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'us_portfolio_optimized.png'), dpi=150)
    plt.show()

print("\n✅ 独立模型训练与回测完成（GOOGL 使用优化模型）")