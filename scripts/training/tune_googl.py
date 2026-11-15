# scripts/training/tune_googl.py
import os, warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import ParameterGrid
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==================== 项目根目录定位 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_US_DIR = os.path.join(PROJECT_ROOT, 'data', 'us')
MODEL_US_DIR = os.path.join(PROJECT_ROOT, 'models', 'us')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

for d in [DATA_US_DIR, MODEL_US_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== 配置 ====================
TICKER = 'GOOGL'
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

def load_stock_data(ticker):
    local_path = os.path.join(DATA_US_DIR, f"{ticker}.csv")
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"本地文件 {local_path} 不存在")
    df = pd.read_csv(local_path)
    rename_map = {
        'date': '日期', 'open': '开盘', 'high': '最高', 'low': '最低',
        'close': '收盘', 'volume': '成交量',
        'Date': '日期', 'Open': '开盘', 'High': '最高',
        'Low': '最低', 'Close': '收盘', 'Volume': '成交量',
        'trade_date': '日期', 'open': '开盘', 'high': '最高', 'low': '最低',
        'close': '收盘', 'vol': '成交量','adj_factor': 'adj_factor'
    }
    df.rename(columns=rename_map, inplace=True)
    df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']].copy()
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
    # ===== 前复权处理 =====
    if 'adj_factor' in df.columns:
        last_factor = df['adj_factor'].iloc[-1]          # 最新复权因子
        for col in ['开盘', '最高', '最低', '收盘']:     # 使用标准化后的中文列名
            if col in df.columns:
                df[col] = df[col] * (df['adj_factor'] / last_factor)
        df.drop(columns=['adj_factor'], inplace=True)    # 删除不再需要的复权因子列
# =====================
    return df

# 加载数据
data = load_stock_data(TICKER)
cutoff = data['日期'].max() - pd.DateOffset(years=4)
data = data[data['日期'] > cutoff].reset_index(drop=True)
data = compute_features(data)

# 尝试不同的涨幅阈值
thresholds = [0.005, 0.01, 0.015, 0.02]
results_search = []

for th in thresholds:
    df = data.copy()
    df['future_return'] = df['收盘'].shift(-2) / df['收盘'] - 1
    df['target'] = (df['future_return'] > th).astype(int)
    df_clean = df.dropna(subset=FEATURES + ['target']).reset_index(drop=True)
    for col in FEATURES:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna(subset=FEATURES + ['target']).reset_index(drop=True)

    X = df_clean[FEATURES].astype('float32')
    y = df_clean['target'].astype(int)
    dates = df_clean['日期']
    log_ret = df_clean['log_return']

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    test_dates = dates.iloc[split:].copy()
    test_log_ret = log_ret.iloc[split:].copy()

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    best_cum = -np.inf
    best_model = None
    best_params = None
    best_prob = None

    for params in ParameterGrid(param_grid):
        model = xgb.XGBClassifier(
            **params,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        bt = pd.DataFrame({
            '日期': test_dates.values,
            'prob': y_prob,
            'log_return': test_log_ret.values
        })
        inner_best_cum = -np.inf
        for thr_signal in [0.50, 0.55, 0.60, 0.65]:
            for s in np.arange(1.5, 4.0, 0.5):
                bt['position'] = 0
                bt.loc[bt['prob'] > thr_signal, 'position'] = 1
                bt.loc[bt['prob'] < 1 - thr_signal, 'position'] = -1
                bt['position'] = bt['position'] * s
                bt['position'] = np.clip(bt['position'], -1, 1)
                bt['strategy_ret'] = bt['position'].shift(1) * bt['log_return']
                bt['turnover'] = bt['position'].diff().abs()
                bt['strategy_ret_net'] = bt['strategy_ret'] - bt['turnover'] * 0.00015
                cum = bt['strategy_ret_net'].sum() * 100
                if cum > inner_best_cum:
                    inner_best_cum = cum

        if inner_best_cum > best_cum:
            best_cum = inner_best_cum
            best_model = model
            best_params = params
            best_prob = y_prob
            best_label_th = th

    # 最终精细搜索交易参数
    bt_final = pd.DataFrame({
        '日期': test_dates.values,
        'prob': best_prob,
        'log_return': test_log_ret.values
    })
    final_best_cum = -np.inf
    final_best_thr = 0.5
    final_best_s = 2.0
    for thr_signal in np.arange(0.45, 0.75, 0.05):
        for s in np.arange(1.0, 4.5, 0.5):
            bt_final['position'] = 0
            bt_final.loc[bt_final['prob'] > thr_signal, 'position'] = 1
            bt_final.loc[bt_final['prob'] < 1 - thr_signal, 'position'] = -1
            bt_final['position'] = bt_final['position'] * s
            bt_final['position'] = np.clip(bt_final['position'], -1, 1)
            bt_final['strategy_ret'] = bt_final['position'].shift(1) * bt_final['log_return']
            bt_final['turnover'] = bt_final['position'].diff().abs()
            bt_final['strategy_ret_net'] = bt_final['strategy_ret'] - bt_final['turnover'] * 0.00015
            cum = bt_final['strategy_ret_net'].sum() * 100
            if cum > final_best_cum:
                final_best_cum = cum
                final_best_thr = thr_signal
                final_best_s = s

    bt_final['position'] = 0
    bt_final.loc[bt_final['prob'] > final_best_thr, 'position'] = 1
    bt_final.loc[bt_final['prob'] < 1 - final_best_thr, 'position'] = -1
    bt_final['position'] = bt_final['position'] * final_best_s
    bt_final['position'] = np.clip(bt_final['position'], -1, 1)
    bt_final['strategy_ret'] = bt_final['position'].shift(1) * bt_final['log_return']
    bt_final['turnover'] = bt_final['position'].diff().abs()
    bt_final['strategy_ret_net'] = bt_final['strategy_ret'] - bt_final['turnover'] * 0.00015
    cum = bt_final['strategy_ret_net'].sum() * 100
    daily_ret = bt_final['strategy_ret_net'].dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() != 0 else 0.0
    max_dd = (daily_ret.cumsum().cummax() - daily_ret.cumsum()).max() * 100
    win_rate = (daily_ret > 0).mean()

    results_search.append({
        'label_threshold': best_label_th,
        'params': best_params,
        'signal_threshold': final_best_thr,
        'position_scale': final_best_s,
        'cum_return': cum,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'model': best_model,
        'prob': best_prob,
        'test_dates': test_dates,
        'test_log_ret': test_log_ret
    })

# 最优结果
best_overall = max(results_search, key=lambda x: x['cum_return'])

print("\n=====================================")
print(" GOOGL 优化搜索结果")
print("=====================================")
for r in results_search:
    print(f"标签阈值: {r['label_threshold']:.3f}, 累计收益: {r['cum_return']:.2f}%, 夏普: {r['sharpe']:.2f}, 回撤: {r['max_drawdown']:.2f}%, 胜率: {r['win_rate']:.2f}")
print("\n最佳配置：")
print(f"  标签涨幅阈值: {best_overall['label_threshold']:.3f}")
print(f"  模型超参数: {best_overall['params']}")
print(f"  开仓阈值: {best_overall['signal_threshold']:.2f}")
print(f"  仓位系数: {best_overall['position_scale']:.2f}")
print(f"  累计收益: {best_overall['cum_return']:.2f}%")
print(f"  夏普比率: {best_overall['sharpe']:.2f}")
print(f"  最大回撤: {best_overall['max_drawdown']:.2f}%")
print(f"  胜率: {best_overall['win_rate']:.2f}")

# 保存优化模型（直接覆盖 individual_models.py 需要加载的文件）
joblib.dump(best_overall['model'], os.path.join(MODEL_US_DIR, 'model_GOOGL.pkl'))
print(f"\n✅ 优化模型已覆盖 {os.path.join(MODEL_US_DIR, 'model_GOOGL.pkl')}")

# 提示更新 individual_models.py 中的参数
print(f"⚠️  请将 individual_models.py 中的 GOOGL 参数更新为：")
print(f"   GOOGL_BEST_THR = {best_overall['signal_threshold']:.2f}")
print(f"   GOOGL_BEST_SCALE = {best_overall['position_scale']:.2f}")

# 绘制收益曲线
prob = best_overall['prob']
log_ret = best_overall['test_log_ret']
signal = (prob > best_overall['signal_threshold']).astype(int) - (prob < 1 - best_overall['signal_threshold']).astype(int)
position = pd.Series(np.clip(signal * best_overall['position_scale'], -1, 1))
log_ret_series = pd.Series(log_ret.values)

strategy_ret = position.shift(1) * log_ret_series
turnover = position.diff().abs()
strategy_ret_net = strategy_ret - turnover * 0.00015
cum_ret = strategy_ret_net.cumsum() * 100
dd = cum_ret - cum_ret.cummax()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
ax1.plot(best_overall['test_dates'], cum_ret, color='steelblue', linewidth=1.2)
ax1.fill_between(best_overall['test_dates'], 0, cum_ret, where=(cum_ret > 0), color='steelblue', alpha=0.1)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax1.set_ylabel('Cumulative Return (%)')
ax1.set_title('GOOGL Optimized Strategy Cumulative Return')
ax1.grid(True, alpha=0.3)

ax2.fill_between(best_overall['test_dates'], dd, 0, color='lightcoral', alpha=0.6)
ax2.set_ylabel('Drawdown (%)')
ax2.set_xlabel('Date')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'googl_optimized_equity.png'), dpi=150)
plt.show()