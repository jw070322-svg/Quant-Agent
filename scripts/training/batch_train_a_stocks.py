# scripts/training/batch_train_a_stocks.py
import os, warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
# 你可以在这里替换成自己的股票池
STOCK_POOL = [
    '603986.SH',
    '603773.SH',
    '300975.SZ',
    # 如果以后还想加入更多，继续往下加即可
]

# 数据目录（请根据你的实际路径修改）
DATA_DIR = r"F:\PythonProject\quant_project\data\cn"
# 模型保存目录
MODEL_DIR = r"F:\PythonProject\quant_project\models\cn"

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

# ==================== 数据加载（兼容 Tushare） ====================
def load_stock_data(ticker):
    local_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"本地文件 {local_path} 不存在")
    df = pd.read_csv(local_path)
    if df.empty:
        raise ValueError(f"{ticker}.csv 为空")
    rename_map = {
        'trade_date': '日期', 'open': '开盘', 'high': '最高',
        'low': '最低', 'close': '收盘', 'vol': '成交量'
    }
    df.rename(columns=rename_map, inplace=True)
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
    # A股 T+1，预测下一天涨幅
    df['future_return'] = df['收盘'].shift(-1) / df['收盘'] - 1
    df['target'] = (df['future_return'] > 0.01).astype(int)
    df_clean = df.dropna(subset=FEATURES + ['target']).reset_index(drop=True)
    for col in FEATURES:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna(subset=FEATURES + ['target']).reset_index(drop=True)
    return df_clean

# ==================== 批量训练与评估 ====================
all_test_returns = []
performance = []

for ticker in STOCK_POOL:
    print(f"\n处理 {ticker}...")
    try:
        data = load_stock_data(ticker)
    except Exception as e:
        print(f"  ❌ 加载失败: {e}，跳过")
        continue

    data = compute_features(data)
    df_clean = prepare_full_data(data)
    n = len(df_clean)
    if n < 500:
        print(f"  ⚠️ 有效样本不足 ({n})，跳过")
        continue

    split = int(n * 0.8)
    train_df = df_clean.iloc[:split]
    test_df = df_clean.iloc[split:]

    X_train = train_df[FEATURES].astype('float32')
    y_train = train_df['target'].astype(int)
    X_test = test_df[FEATURES].astype('float32')
    y_test = test_df['target'].astype(int)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)

    test = test_df.copy()
    test['prob'] = y_prob
    test['position'] = np.where(test['prob'] > 0.5, 1, 0)
    test['ret'] = test['position'].shift(1) * test['log_return']
    test['turnover'] = test['position'].diff().abs()
    # A股成本：单边约 0.1% (印花税+佣金)
    test['ret_net'] = test['ret'] - test['turnover'] * 0.001
    cum = test['ret_net'].sum() * 100
    sharpe = (test['ret_net'].mean() / test['ret_net'].std()) * np.sqrt(252) if test['ret_net'].std() != 0 else 0.0
    max_dd = (test['ret_net'].cumsum().cummax() - test['ret_net'].cumsum()).max() * 100
    win_rate = (test.loc[test['position'].shift(1) == 1, 'ret_net'] > 0).mean() if len(test[test['position'].shift(1) == 1]) > 0 else 0.0

    performance.append({
        '代码': ticker,
        '累计收益(%)': round(cum, 2),
        '夏普比率': round(sharpe, 2),
        '最大回撤(%)': round(max_dd, 2),
        '胜率': round(win_rate, 4),
        '准确率': round(acc, 4)
    })

    # 保存模型和特征到 MODEL_DIR
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, f'model_{ticker}.pkl'))
    with open(os.path.join(MODEL_DIR, f'features_{ticker}.txt'), 'w') as f:
        f.write(','.join(FEATURES))

    ret_series = test.set_index('日期')['ret_net']
    ret_series.name = ticker
    all_test_returns.append(ret_series)

    print(f"  样本: {n}, 训练集: {len(train_df)}, 测试集: {len(test_df)}")
    print(f"  准确率: {acc:.4f}, 累计收益: {cum:.2f}%, 夏普: {sharpe:.2f}, 最大回撤: {max_dd:.2f}%")

# ==================== 等权组合 ====================
if all_test_returns:
    combined = pd.concat(all_test_returns, axis=1).dropna()
    if not combined.empty:
        port_ret = combined.mean(axis=1)
        port_cum = port_ret.cumsum() * 100
        port_sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(252) if port_ret.std() != 0 else 0.0
        port_dd = (port_cum.cummax() - port_cum).max()
        port_win = (port_ret > 0).mean()
        print("\n===== 等权组合绩效 =====")
        print(f"组合累计收益: {port_cum.iloc[-1]:.2f}%")
        print(f"组合夏普比率: {port_sharpe:.2f}")
        print(f"组合最大回撤: {port_dd:.2f}%")
        print(f"组合胜率: {port_win:.4f}")

# ==================== 绩效报告 ====================
print("\n" + "=" * 80)
print("                         📈 A股批量训练绩效报告")
print("=" * 80)
header = f"{'代码':<12} {'累计收益%':>10} {'夏普':>8} {'最大回撤%':>10} {'胜率':>8} {'准确率':>8}"
print(header)
print("-" * 80)
for p in performance:
    print(f"{p['代码']:<12} {p['累计收益(%)']:>10.02f} {p['夏普比率']:>8.02f} {p['最大回撤(%)']:>10.02f} {p['胜率']:>8.04f} {p['准确率']:>8.04f}")
print("-" * 80)
if all_test_returns and 'port_cum' in dir():
    print(f"  等权组合累计收益: {port_cum.iloc[-1]:.2f}%")
    print(f"  等权组合夏普比率: {port_sharpe:.2f}")
    print(f"  等权组合最大回撤: {port_dd:.2f}%")
    print(f"  等权组合胜率: {port_win:.4f}")
print("=" * 80)

# 可选：保存组合收益图
if all_test_returns and 'port_cum' in dir():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(port_cum.index, port_cum, color='steelblue', linewidth=1.2, label='等权组合')
    ax1.fill_between(port_cum.index, 0, port_cum, where=(port_cum > 0), color='steelblue', alpha=0.1)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.set_title('A-Share Equal-Weight Portfolio Performance')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    dd = port_cum - port_cum.cummax()
    ax2.fill_between(port_cum.index, dd, 0, color='lightcoral', alpha=0.6)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, '..', '..', 'outputs', 'a_stock_portfolio.png'), dpi=150)
    plt.show()

print("\n✅ 批量训练完成")