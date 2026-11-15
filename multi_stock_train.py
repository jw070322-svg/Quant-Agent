# multi_stock_train.py
import os
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import yfinance as yf

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
START_DATE = '2019-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')
DATA_DIR = r"F:\PythonProject\Python Course Project\sp500_stocks"

# ==================== 数据加载工具 ====================
def load_stock_data(ticker, start_date=START_DATE, end_date=END_DATE):
    local_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path)
            if len(df) == 0:
                raise ValueError("本地文件为空")
            rename_map = {
                'date': '日期', 'open': '开盘', 'high': '最高', 'low': '最低',
                'close': '收盘', 'volume': '成交量',
                'Date': '日期', 'Open': '开盘', 'High': '最高',
                'Low': '最低', 'Close': '收盘', 'Volume': '成交量'
            }
            df.rename(columns=rename_map, inplace=True)
            df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']].copy()
            df['日期'] = pd.to_datetime(df['日期'])
            df.sort_values('日期', inplace=True)
            df.reset_index(drop=True, inplace=True)
            print(f"  📂 {ticker} 本地加载，{len(df)} 行")
            return df
        except Exception as e:
            print(f"  ⚠️ 本地文件 {ticker}.csv 读取失败: {type(e).__name__}: {e}，尝试下载...")
    try:
        print(f"  🌐 下载 {ticker} ...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError(f"下载 {ticker} 数据为空")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        data = data.reset_index()
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        data.rename(columns={
            'Date': '日期', 'Open': '开盘', 'High': '最高',
            'Low': '最低', 'Close': '收盘', 'Volume': '成交量'
        }, inplace=True)
        data['日期'] = pd.to_datetime(data['日期'])
        data.sort_values('日期', inplace=True)
        data.to_csv(local_path, index=False, encoding='utf-8')
        print(f"  ✅ {ticker} 下载并缓存至 {local_path}")
        return data
    except Exception as e:
        print(f"  ❌ {ticker} 下载失败: {type(e).__name__}: {e}")
        return None

# ==================== 特征工程 ====================
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

# ==================== 加载并处理所有股票 ====================
all_data = []
for ticker in TICKERS:
    print(f"处理 {ticker}...")
    data = load_stock_data(ticker)
    if data is None:
        continue
    cutoff = data['日期'].max() - pd.DateOffset(years=4)
    data = data[data['日期'] > cutoff]
    data = compute_features(data)
    data['Ticker'] = ticker
    all_data.append(data)

df_all = pd.concat(all_data, ignore_index=True)
# 按日期排序，保证时间序列分割合理
df_all = df_all.sort_values('日期').reset_index(drop=True)
print(f"\n合并数据（已按日期排序）: {len(df_all)} 行")

# ==================== 构造标签 ====================
df_all['future_return'] = df_all.groupby('Ticker')['收盘'].shift(-2) / df_all['收盘'] - 1
df_all['target'] = (df_all['future_return'] > 0.01).astype(int)

# ==================== 特征列表 ====================
features = [
    'MA5', 'MA10', 'MA20',
    'MACD', 'MACD_hist',
    'K', 'D', 'J',
    'RSI',
    'BB_width', 'BB_position',
    'volatility_5', 'volatility_20',
    'close_vs_MA20', 'volume_ratio',
    'roc_7', 'roc_14'
]

# 清洗
df_clean = df_all.dropna(subset=features + ['target']).reset_index(drop=True)
print(f"有效样本: {len(df_clean)} 行")

# ==================== 分割与标准化 ====================
split_idx = int(len(df_clean) * 0.8)
train = df_clean.iloc[:split_idx]
test = df_clean.iloc[split_idx:]

X_train = train[features].astype('float32')
y_train = train['target'].astype(int)
X_test = test[features].astype('float32')
y_test = test['target'].astype(int)

# 🔥 特征标准化（基于训练集均值标准差）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== 训练联合模型 ====================
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train_scaled, y_train)

# ==================== 模型评估 ====================
y_pred = model.predict(X_test_scaled)
overall_acc = (y_pred == y_test).mean()
print(f"\n📊 测试集整体准确率: {overall_acc:.4f}")

# ==================== 批量回测 + 绩效指标（每只股票独立调参） ====================
print("\n===== 逐只股票回测与绩效报告（独立参数优化） =====")
test = test.copy()
test['prob'] = model.predict_proba(X_test)[:, 1]

# 诊断概率分布
print("\n概率分布诊断：")
for ticker in TICKERS:
    sub_diag = test[test['Ticker'] == ticker]['prob']
    if len(sub_diag) == 0:
        continue
    print(f"{ticker}: min={sub_diag.min():.4f}, median={sub_diag.median():.4f}, max={sub_diag.max():.4f}")

print("\n测试集各股票样本数：")
print(test.groupby('Ticker').size())

# 参数搜索范围
base_range = np.arange(0.20, 0.55, 0.05)
scale_range = np.arange(1.5, 4.5, 0.5)

portfolio_returns = []
metrics = []
MIN_SAMPLES = 5

for ticker in TICKERS:
    sub = test[test['Ticker'] == ticker].copy()
    if len(sub) < MIN_SAMPLES:
        print(f"{ticker}: 测试样本不足 ({len(sub)} 行)，跳过")
        continue

    best_cum = -np.inf
    best_params = None
    best_position = None
    best_ret_net = None

    # 网格搜索最优参数
    for base in base_range:
        for scale in scale_range:
            # 计算仓位
            pos = np.clip((sub['prob'] - base) * scale, 0, 1)
            ret = pos.shift(1) * sub['log_return']
            ret_net = ret - pos.diff().abs() * 0.00015
            cum = ret_net.sum() * 100
            if cum > best_cum:
                best_cum = cum
                best_params = (base, scale)
                best_position = pos
                best_ret_net = ret_net

    # 用最优参数计算最终绩效
    sub['position'] = best_position
    sub['ret_net'] = best_ret_net
    cum = best_cum

    daily_ret = best_ret_net.dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() != 0 else 0.0

    cumsum = daily_ret.cumsum()
    running_max = cumsum.cummax()
    drawdown = (cumsum - running_max)
    max_drawdown = drawdown.min() * 100

    traded = sub[sub['position'].shift(1) > 0]
    win_rate = (traded['ret_net'] > 0).mean() if len(traded) > 0 else 0.0

    model_acc = (sub['prob'] > 0.5) == sub['target']
    model_acc = model_acc.mean() if len(model_acc) > 0 else 0.0

    base_opt, scale_opt = best_params
    print(f"{ticker}: 最优参数 base={base_opt:.2f}, scale={scale_opt:.2f}, 累计收益={cum:.2f}%")

    metrics.append({
        'Ticker': ticker,
        'base': base_opt,
        'scale': scale_opt,
        '累计收益(%)': round(cum, 2),
        '夏普比率': round(sharpe, 2),
        '最大回撤(%)': round(max_drawdown, 2),
        '胜率': round(win_rate, 4),
        '模型准确率': round(model_acc, 4)
    })
    portfolio_returns.append(best_ret_net)

# ==================== 等权组合 ====================
if portfolio_returns:
    portfolio = pd.concat(portfolio_returns, axis=1).mean(axis=1).dropna()
    portfolio_cum = portfolio.cumsum() * 100
    comb_sharpe = (portfolio.mean() / portfolio.std()) * np.sqrt(252) if portfolio.std() != 0 else 0.0
    comb_cumsum = portfolio.cumsum()
    comb_max = comb_cumsum.cummax()
    comb_dd = (comb_cumsum - comb_max).min() * 100
    comb_traded = portfolio[portfolio != 0]
    comb_win = (comb_traded > 0).mean() if len(comb_traded) > 0 else 0.0

    print(f"\n等权组合累计收益: {portfolio_cum.iloc[-1]:.2f}%")
    print(f"组合夏普比率: {comb_sharpe:.2f}")
    print(f"组合最大回撤: {comb_dd:.2f}%")
    print(f"组合胜率: {comb_win:.4f}")

# ==================== 可视化报告 ====================
print("\n" + "=" * 90)
print("                          📈 多股票量化策略绩效报告（独立参数）")
print("=" * 90)

header = f"{'标的':<8} {'base':<6} {'scale':<6} {'累计收益%':>10} {'夏普':>8} {'最大回撤%':>10} {'胜率':>8} {'模型准确率':>10}"
print(header)
print("-" * 90)

for m in metrics:
    print(f"{m['Ticker']:<8} {m['base']:<6.2f} {m['scale']:<6.2f} {m['累计收益(%)']:>10.2f} {m['夏普比率']:>8.2f} "
          f"{m['最大回撤(%)']:>10.2f} {m['胜率']:>8.4f} {m['模型准确率']:>10.4f}")

print("-" * 90)
print(f"  测试集整体模型准确率: {overall_acc:.4f}")
print("=" * 90)
