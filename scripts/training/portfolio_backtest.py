# scripts/validation/portfolio_backtest.py
import os, warnings
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'us')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'us')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

TICKERS = ['AAPL', 'AMZN', 'GOOGL']
FEATURES = ['MA5', 'MA10', 'MA20', 'MACD', 'MACD_hist',
            'K', 'D', 'J', 'RSI', 'BB_width', 'BB_position',
            'volatility_5', 'volatility_20', 'close_vs_MA20',
            'volume_ratio', 'roc_7', 'roc_14']

# 每只股票预存的交易参数（来自各自的最优搜索）
BEST_PARAMS = {
    'AAPL':  {'threshold': 0.50, 'scale': 1.5},
    'AMZN':  {'threshold': 0.50, 'scale': 1.5},
    'GOOGL': {'threshold': 0.65, 'scale': 1.0}   # 来自 tune_googl 的最优结果
}

def load_stock_data(ticker):
    p = os.path.join(DATA_DIR, f"{ticker}.csv")
    df = pd.read_csv(p)
    rename = {'date':'日期','open':'开盘','high':'最高','low':'最低','close':'收盘','volume':'成交量',
              'Date':'日期','Open':'开盘','High':'最高','Low':'最低','Close':'收盘','Volume':'成交量'}
    df.rename(columns=rename, inplace=True)
    df = df[['日期','开盘','最高','最低','收盘','成交量']]
    df['日期'] = pd.to_datetime(df['日期'])
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

def prepare_data(df):
    df = df.copy()
    df['future_return'] = df['收盘'].shift(-2) / df['收盘'] - 1
    df['target'] = (df['future_return'] > 0.01).astype(int)
    df_clean = df.dropna(subset=FEATURES + ['target']).reset_index(drop=True)
    for c in FEATURES:
        df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')
    df_clean = df_clean.dropna(subset=FEATURES + ['target']).reset_index(drop=True)
    X = df_clean[FEATURES].astype('float32')
    y = df_clean['target'].astype(int)
    dates = df_clean['日期']
    log_ret = df_clean['log_return']
    return X, y, dates, log_ret

portfolio_returns = []
performance = []

for ticker in TICKERS:
    print(f"处理 {ticker}...")
    data = load_stock_data(ticker)
    cutoff = data['日期'].max() - pd.DateOffset(years=4)
    data = data[data['日期'] > cutoff].reset_index(drop=True)
    data = compute_features(data)
    X, y, dates, log_ret = prepare_data(data)

    model_path = os.path.join(MODEL_DIR, f'model_{ticker}.pkl')
    if not os.path.exists(model_path):
        print(f"  模型文件 {model_path} 不存在，跳过")
        continue
    model = joblib.load(model_path)

    split = int(len(X) * 0.8)
    X_test = X[split:]
    test_dates = dates.iloc[split:].copy()
    test_log_ret = log_ret.iloc[split:].copy()
    y_test = y.iloc[split:]

    y_prob = model.predict_proba(X_test)[:, 1]

    thr = BEST_PARAMS[ticker]['threshold']
    s = BEST_PARAMS[ticker]['scale']

    bt = pd.DataFrame({'日期': test_dates.values, 'prob': y_prob, 'log_return': test_log_ret.values})
    bt['position'] = 0
    bt.loc[bt['prob'] > thr, 'position'] = 1
    bt.loc[bt['prob'] < 1 - thr, 'position'] = -1
    bt['position'] = bt['position'] * s
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

    performance.append({
        'Ticker': ticker,
        '累计收益(%)': round(cum, 2),
        '夏普比率': round(sharpe, 2),
        '最大回撤(%)': round(max_dd, 2),
        '胜率': round(win_rate, 4)
    })

    ret_series = bt.set_index('日期')['strategy_ret_net']
    ret_series.name = ticker
    portfolio_returns.append(ret_series)
    print(f"  {ticker}: 收益 {cum:.2f}%, 夏普 {sharpe:.2f}")

# 等权组合
if portfolio_returns:
    combined = pd.concat(portfolio_returns, axis=1).dropna()
    port_ret = combined.mean(axis=1)
    port_cum = port_ret.cumsum() * 100
    port_sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(252) if port_ret.std() != 0 else 0.0
    port_dd = (port_cum.cummax() - port_cum).max()
    port_win = (port_ret > 0).mean()

    print("\n===== 等权组合（优化后） =====")
    print(f"组合累计收益: {port_cum.iloc[-1]:.2f}%")
    print(f"组合夏普比率: {port_sharpe:.2f}")
    print(f"组合最大回撤: {port_dd:.2f}%")
    print(f"组合胜率: {port_win:.4f}")

# 绩效表
print("\n" + "=" * 90)
print("                     📈 优化后美股组合绩效报告")
print("=" * 90)
header = f"{'标的':<8} {'累计收益%':>10} {'夏普':>8} {'最大回撤%':>10} {'胜率':>8}"
print(header)
print("-" * 90)
for p in performance:
    print(f"{p['Ticker']:<8} {p['累计收益(%)']:>10.2f} {p['夏普比率']:>8.2f} {p['最大回撤(%)']:>10.2f} {p['胜率']:>8.4f}")
print("-" * 90)
if portfolio_returns:
    print(f"  等权组合累计收益: {port_cum.iloc[-1]:.2f}%")
    print(f"  等权组合夏普比率: {port_sharpe:.2f}")
    print(f"  等权组合最大回撤: {port_dd:.2f}%")
    print(f"  等权组合胜率: {port_win:.4f}")
print("=" * 90)

# 保存图表
if portfolio_returns:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(port_cum.index, port_cum, color='steelblue')
    ax.set_title('Optimized Portfolio Cumulative Return')
    ax.set_ylabel('Cumulative Return (%)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'optimized_portfolio.png'), dpi=150)
    plt.show()

print("\n✅ 优化组合回测完成")