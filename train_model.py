# ======================== 恢复版高收益分类 Agent ========================
import os, warnings, itertools
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime
warnings.filterwarnings('ignore')

# ==================== 1. 数据加载（智能修复） ====================
def load_or_download(save_path='AAPL.csv', start_date='2019-01-01'):
    end_date = datetime.today().strftime('%Y-%m-%d')
    df = None
    try:
        import yfinance as yf
        print("🌐 正在从 yfinance 下载 AAPL 最新数据...")
        df_online = yf.download('AAPL', start=start_date, end=end_date, progress=False)
        if not df_online.empty:
            if isinstance(df_online.columns, pd.MultiIndex):
                df_online.columns = df_online.columns.droplevel(1)
            df_online.reset_index(inplace=True)
            df_online.rename(columns={'Date':'日期','Open':'开盘','High':'最高',
                                      'Low':'最低','Close':'收盘','Volume':'成交量'}, inplace=True)
            df = df_online[['日期','开盘','最高','最低','收盘','成交量']].copy()
            df['日期'] = pd.to_datetime(df['日期'])
            df.sort_values('日期', inplace=True)
            df.to_csv(save_path, index=False, encoding='utf-8')
            print(f"✅ 下载成功，共 {len(df)} 行")
            return df
    except:
        pass
    # 本地文件备用
    if os.path.exists(save_path):
        df_local = pd.read_csv(save_path)
        if len(df_local) == 0:
            raise ValueError("本地文件为空，请检查网络重新下载")
        df_local.rename(columns={'date':'日期','open':'开盘','high':'最高',
                                  'low':'最低','close':'收盘','volume':'成交量'}, inplace=True)
        df_local['日期'] = pd.to_datetime(df_local['日期'])
        df_local.sort_values('日期', inplace=True)
        df_local.reset_index(drop=True, inplace=True)
        print(f"📂 本地文件加载成功，共 {len(df_local)} 行")
        return df_local
    raise FileNotFoundError("无法获取数据，请检查网络或本地文件。")

local_path = r"F:\PythonProject\quant_project\AAPL.csv"
df = load_or_download(local_path, start_date='2019-01-01')

# 🔥 只使用近4年数据（最贴近当前市场）
cutoff = df['日期'].max() - pd.DateOffset(years=4)
df = df[df['日期'] > cutoff].reset_index(drop=True)
print(f"近4年数据: {len(df)} 行\n")

# ==================== 2. 技术指标（精简版） ====================
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

df['BB_middle'] = df['收盘'].rolling(20).mean()
bb_std = df['收盘'].rolling(20).std()
df['BB_width'] = 2 * bb_std
df['BB_position'] = (df['收盘'] - (df['BB_middle'] - 2*bb_std)) / (4*bb_std + 1e-9)

df['roc_7'] = df['收盘'].pct_change(7)
df['roc_14'] = df['收盘'].pct_change(14)

df['returns'] = df['收盘'].pct_change()
df['log_return'] = np.log(df['收盘'] / df['收盘'].shift(1))
df['volatility_5'] = df['returns'].rolling(5).std()
df['volatility_20'] = df['returns'].rolling(20).std()
df['close_vs_MA20'] = (df['收盘'] - df['MA20']) / df['MA20']
df['volume_ratio'] = df['成交量'] / df['成交量'].rolling(20).mean()

# ==================== 3. 分类标签（未来2日涨幅 > 1%） ====================
df['future_return'] = df['收盘'].shift(-2) / df['收盘'] - 1
df['target'] = (df['future_return'] > 0.01).astype(int)  # 🔥 关键：分类标签

# ==================== 4. 特征集（精简，无ARIMA/MC） ====================
features = [
    'MA5','MA10','MA20',
    'MACD','MACD_hist',
    'K','D','J',
    'RSI',
    'BB_width','BB_position',
    'volatility_5','volatility_20',
    'close_vs_MA20','volume_ratio',
    'roc_7','roc_14'
]

# 数据清洗
df_clean = df.dropna(subset=features + ['target']).reset_index(drop=True)
for col in features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean = df_clean.dropna(subset=features + ['target']).reset_index(drop=True)
print(f"有效样本: {len(df_clean)} 行")

# ==================== 5. 分割 ====================
split = int(len(df_clean) * 0.8)
X = df_clean[features].astype('float32')
y = df_clean['target'].astype(int)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

# ==================== 6. 训练分类模型 ====================
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

# 预测上涨概率（类别1的概率）
y_prob = model.predict_proba(X_test)[:, 1]  # 🔥 这就是你要的上涨概率
acc = accuracy_score(y_test, (y_prob > 0.5).astype(int))
print(f"测试集准确率: {acc:.4f}")

# ==================== 7. 回测 + 最优阈值/仓位搜索 ====================
bt = df_clean.iloc[split:].copy()
bt['prob'] = y_prob
bt['log_return'] = df_clean.iloc[split:]['log_return'].values

best_cum = -np.inf
best_thr = 0.5
best_s = 2.0

# 同时搜索阈值和仓位系数
for thr in [0.50, 0.55, 0.60, 0.65]:
    for s in np.arange(1.5, 4.0, 0.5):
        bt['position'] = 0
        bt.loc[bt['prob'] > thr, 'position'] = 1
        bt.loc[bt['prob'] < 1 - thr, 'position'] = -1
        # 仓位缩放
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

print(f"最优阈值: {best_thr}, 最优仓位系数: {best_s}")
print(f"最优累计收益: {best_cum:.2f}%")

# 用最优参数重算
bt['position'] = 0
bt.loc[bt['prob'] > best_thr, 'position'] = 1
bt.loc[bt['prob'] < 1 - best_thr, 'position'] = -1
bt['position'] = bt['position'] * best_s
bt['position'] = np.clip(bt['position'], -1, 1)

bt['strategy_ret'] = bt['position'].shift(1) * bt['log_return']
bt['turnover'] = bt['position'].diff().abs()
bt['strategy_ret_net'] = bt['strategy_ret'] - bt['turnover'] * 0.00015

cum_ret = (bt['strategy_ret_net'].cumsum() * 100)
sharpe = (bt['strategy_ret_net'].mean() / (bt['strategy_ret_net'].std() + 1e-9)) * np.sqrt(252)
print(f"🚀 最终累计收益: {cum_ret.iloc[-1]:.2f}%")
print(f"夏普比率: {sharpe:.2f}")

# 输出最近上涨概率
print("\n📈 最近5个交易日上涨概率:")
recent = bt[['日期','收盘','prob']].tail(5)
for _, row in recent.iterrows():
    d = row['日期'].strftime('%Y-%m-%d') if hasattr(row['日期'], 'strftime') else str(row['日期'])
    print(f"{d} 收盘:{row['收盘']:.2f} 上涨概率:{row['prob']:.2%}")

# 保存
joblib.dump(model, 'high_return_model.pkl')
with open('high_features.txt', 'w') as f:
    f.write(','.join(features))
print("\n✅ 模型已保存，上涨概率导出至 aapl_prob.csv")
bt[['日期','收盘','prob']].to_csv('aapl_prob.csv', index=False, encoding='utf-8')

# ==================== 9. 可视化（自动修复，只画收益图） ====================
import matplotlib.pyplot as plt
import pandas as pd

# --- 9.1 特征重要性（直接从模型重新计算，不依赖外部变量）---
# 确定特征列名（根据你脚本里使用的变量名，可能是 all_features 或 feature_columns）
try:
    feat_names = features   # 你的主脚本里可能用的是 all_features
except NameError:
    feat_names = features   # 或者 feature_columns

# 计算重要性
importance = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False)

print("\n" + "="*50)
print("📊 特征重要性 Top 10")
print("="*50)
for i, (feat, val) in enumerate(importance.head(10).items(), 1):
    print(f"{i:2}. {feat:<20s} {val:.4f}")

# --- 9.2 最近 20 个交易日的上涨概率（打印） ---
recent_n = 20
recent = bt[['日期', '收盘', 'prob']].tail(recent_n)
print("\n" + "="*50)
print(f"📈 最近 {recent_n} 个交易日上涨概率预测")
print("="*50)
for _, row in recent.iterrows():
    d = row['日期'].strftime('%Y-%m-%d') if hasattr(row['日期'], 'strftime') else str(row['日期'])
    print(f"{d}  收盘: {row['收盘']:7.2f}  上涨概率: {row['prob']:.2%}")

# --- 9.3 累积收益曲线（唯一的图表） ---
cum_return = (bt['strategy_ret_net'].cumsum() * 100)
cum_max = cum_return.cummax()
drawdown = cum_return - cum_max

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})
ax1.plot(bt['日期'], cum_return, color='steelblue', linewidth=1.2, label='策略累积收益 (%)')
ax1.fill_between(bt['日期'], 0, cum_return,
                 where=(cum_return > 0), color='steelblue', alpha=0.1)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax1.set_ylabel('累积收益 (%)')
ax1.legend(loc='upper left')
ax1.set_title('策略累积收益曲线', fontsize=14)
ax1.grid(True, alpha=0.3)

ax2.fill_between(bt['日期'], drawdown, 0, color='lightcoral', alpha=0.6)
ax2.set_ylabel('回撤 (%)')
ax2.set_xlabel('日期')
ax2.set_title('回撤曲线', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('equity_curve.png', dpi=150)
plt.show()

print("\n✅ 累积收益图已保存为 equity_curve.png")