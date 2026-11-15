# scripts/live/trade_engine.py
import os, json, warnings
import pandas as pd
import numpy as np
from portfolio_optimizer import optimize_portfolio
from datetime import datetime

warnings.filterwarnings('ignore')

# ==================== 项目根目录定位 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'cn')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'cn')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# ==================== 配置 ====================
import json

# 尝试读取选股结果
selected_path = os.path.join(OUTPUT_DIR, 'selected_stocks.json')
if os.path.exists(selected_path):
    with open(selected_path, 'r') as f:
        sel = json.load(f)
    QUANT_STOCK = sel['selected'][0] if len(sel['selected']) > 0 else '603986.SH'
    SUBJECTIVE_STOCKS = sel['selected'][1:] if len(sel['selected']) > 1 else ['603773.SH', '300975.SZ']
else:
    QUANT_STOCK = '603986.SH'
    SUBJECTIVE_STOCKS = ['603773.SH', '300975.SZ']

TOTAL_CAPITAL = 1_000_000
QUANT_CAPITAL = 400_000
SUB_CAPITAL_PER_STOCK = 300_000

STOP_LOSS = -0.08
TRAILING_STOP = 0.08
PROFIT_TAKE = 0.15

highest_price = {stock: 0.0 for stock in SUBJECTIVE_STOCKS}

# ==================== 数据加载与特征计算 ====================
def load_recent_data(ticker, days=120):
    path = os.path.join(DATA_DIR, f'{ticker}.csv')
    df = pd.read_csv(path)
    rename_map = {
        'trade_date': '日期', 'open': '开盘', 'high': '最高',
        'low': '最低', 'close': '收盘', 'vol': '成交量'
    }
    df.rename(columns=rename_map, inplace=True)
    if df['日期'].dtype in ('int64', 'float64'):
        df['日期'] = pd.to_datetime(df['日期'].astype(str), format='%Y%m%d')
    else:
        df['日期'] = pd.to_datetime(df['日期'])
    df.sort_values('日期', inplace=True)
    df = df.tail(days).reset_index(drop=True)
    # 前复权处理
    if 'adj_factor' in df.columns:
        last_factor = df['adj_factor'].iloc[-1]
        for col in ['开盘', '最高', '最低', '收盘']:
            if col in df.columns:
                df[col] = df[col] * (df['adj_factor'] / last_factor)
        df.drop(columns=['adj_factor'], inplace=True)
    return df

def compute_indicators(df):
    close = df['收盘'].values
    high = df['最高'].values
    low = df['最低'].values
    volume = df['成交量'].values

    ma5 = float(close[-5:].mean())
    ma10 = float(close[-10:].mean())
    ma20 = float(close[-20:].mean())
    prev_ma5 = float(close[-6:-1].mean())
    prev_ma20 = float(close[-21:-1].mean())

    ema12 = float(close[-12:].mean())
    ema26 = float(close[-26:].mean())
    dif = ema12 - ema26
    dea = float(close[-9:].mean())
    prev_ema12 = float(close[-13:-1].mean())
    prev_ema26 = float(close[-27:-1].mean())
    prev_dif = prev_ema12 - prev_ema26
    prev_dea = float(close[-10:-1].mean())

    high_9 = float(high[-9:].max())
    low_9 = float(low[-9:].min())
    rsv = (close[-1] - low_9) / (high_9 - low_9) * 100.0 if (high_9 - low_9) != 0 else 50.0
    slowk = float(rsv)
    slowd = float(rsv)

    delta = close[-15:] - close[-16:-1]
    gain = float(delta[delta > 0].sum())
    loss = float(-delta[delta < 0].sum())
    rs = gain / loss if loss != 0 else 0.0
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # 20日波动率（修正：使用正确变量）
    returns = np.diff(close) / close[:-1]   # 简单日收益率
    volatility_20 = float(np.std(returns[-20:])) if len(returns) >= 20 else 0.02

    trend_up = bool(ma20 > prev_ma20)

    buy_signal = 0
    if (ma5 > ma20 and prev_ma5 <= prev_ma20) or (ma5 > ma20 and close[-1] > ma5):
        buy_signal += 1
    if dif > dea and prev_dif <= prev_dea:
        buy_signal += 1
    if slowk > slowd and slowk < 80:
        buy_signal += 1
    if 30 < rsi < 70:
        buy_signal += 1

    sell_signal = 0
    if (ma5 < ma20 and prev_ma5 >= prev_ma20) or (close[-1] < ma20):
        sell_signal += 1
    if dif < dea and prev_dif >= prev_dea:
        sell_signal += 1
    if rsi > 75:
        sell_signal += 1

    return {
        'close': float(close[-1]),
        'ma5': ma5, 'ma10': ma10, 'ma20': ma20,
        'prev_ma20': prev_ma20,
        'dif': float(dif), 'dea': float(dea),
        'slowk': slowk, 'slowd': slowd,
        'rsi': float(rsi),
        'trend_up': trend_up,
        'buy_signal': int(buy_signal),
        'sell_signal': int(sell_signal),
        'volume_ratio': float(volume[-1] / volume[-20:].mean()) if volume[-20:].mean() != 0 else 1.0,
        'volatility_20': volatility_20
    }


# ==================== 量化信号 ====================
def get_quant_signal():
    df = load_recent_data(QUANT_STOCK)
    ind = compute_indicators(df)

    threshold = 2
    trend_up = ind['trend_up']
    buy_signal = ind['buy_signal']
    sell_signal = ind['sell_signal']

    signal = "HOLD"
    reason = ""
    if buy_signal >= threshold and trend_up:
        signal = "BUY"
        reason = f"共振={buy_signal}, MA20向上"
    elif sell_signal >= 1:
        signal = "SELL"
        reason = f"卖出信号={sell_signal}"
    else:
        reason = f"共振={buy_signal}, 趋势{'向上' if trend_up else '向下'}"

    return {
        "ticker": QUANT_STOCK,
        "date": str(df.iloc[-1]['日期'].date()),
        "close": round(ind['close'], 2),
        "indicators": {
            "MA5": round(ind['ma5'], 2),
            "MA20": round(ind['ma20'], 2),
            "MACD_DIF": round(ind['dif'], 4),
            "KDJ_K": round(ind['slowk'], 2),
            "RSI": round(ind['rsi'], 2),
            "trend_up": trend_up
        },
        "buy_signal": buy_signal,
        "sell_signal": sell_signal,
        "signal": signal,
        "reason": reason,
        "allocation": QUANT_CAPITAL
    }

# ==================== 主观趋势网格信号 ====================
def get_subjective_signal():
    signals = []
    for ticker in SUBJECTIVE_STOCKS:
        df = load_recent_data(ticker)
        ind = compute_indicators(df)

        close = ind['close']
        ma20 = ind['ma20']
        trend_up = ind['trend_up']
        pullback = bool(close < ma20 * 1.02)

        current_pct = 0.0   # 模拟

        can_buy = trend_up and pullback
        action = "HOLD"
        reason = ""

        if can_buy and current_pct < SUB_CAPITAL_PER_STOCK / TOTAL_CAPITAL:
            action = "BUY"
            if current_pct == 0.0:
                reason = f"首次建仓（趋势向上，回调至MA20={ma20:.2f}附近）"
            else:
                reason = f"加仓（趋势向上，回调加仓）"
        else:
            reason = f"趋势{'向上' if trend_up else '向下'}, 价格{'在MA20附近' if pullback else '高于MA20'}"

        if close > highest_price[ticker]:
            highest_price[ticker] = close

        signals.append({
            "ticker": ticker,
            "date": str(df.iloc[-1]['日期'].date()),
            "close": round(close, 2),
            "ma20": round(ma20, 2),
            "trend_up": trend_up,
            "pullback": pullback,
            "action": action,
            "reason": reason,
            "highest_price": round(highest_price[ticker], 2),
            "trailing_stop_level": round(highest_price[ticker] * (1 - TRAILING_STOP), 2),
            "allocation": SUB_CAPITAL_PER_STOCK
        })

    return signals

# ==================== 自定义 JSON 编码器 ====================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

# ==================== 主函数 ====================
def run():
    # 获取各策略信号
    quant = get_quant_signal()
    subjective = get_subjective_signal()

    # 收集所有候选买入股票及其概率和波动率
    candidates = []
    
    # 量化部分
    if quant['signal'] == 'BUY':
        # buy_signal 值映射为概率代理（0-4 -> 0-1）
        prob = quant['buy_signal'] / 4.0
        vol = quant['indicators'].get('volatility_20', 0.02)
        candidates.append((quant['ticker'], prob, vol))
    
    # 主观部分
    for sub in subjective:
        if sub['action'] == 'BUY':
            # 主观信号没有明确概率，给予 0.6 固定值
            prob = 0.6
            vol = 0.02  # 默认波动率
            candidates.append((sub['ticker'], prob, vol))
    
    # 调用优化器
    orders = []
    if candidates:
        tickers, probs, vols = zip(*candidates)
        try:
            from portfolio_optimizer import optimize_portfolio
            weights = optimize_portfolio(probs, vols)
        except ImportError:
            # 如果优化器不存在，等权分配
            weights = [1.0 / len(candidates)] * len(candidates)
        
        for i, (ticker, prob, vol) in enumerate(candidates):
            orders.append({
                "ticker": ticker,
                "probability_proxy": round(prob, 4),
                "weight": round(weights[i], 4),
                "amount": round(weights[i] * TOTAL_CAPITAL, 2),
                "note": "优化器分配" if 'optimize_portfolio' in dir() else "等权分配"
            })
    else:
        orders = []
    
    # 构建最终决策
    decision = {
        "timestamp": str(datetime.now()),
        "quant": quant,
        "subjective": subjective,
        "optimized_orders": orders,
        "capital": TOTAL_CAPITAL,
        "rules": {
            "quant_stock": QUANT_STOCK,
            "subjective_stocks": SUBJECTIVE_STOCKS,
            "quant_strategy": "多指标共振(>=2)+MA20趋势过滤，止损-8%",
            "subjective_strategy": "趋势回调加仓+移动止盈(回撤8%)+固定止盈15%，止损-8%",
            "portfolio_optimization": "线性规划最大化概率，约束单票≤20%、总仓≤80%、组合波动上限"
        }
    }

    # 保存输出
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'today_decision.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(decision, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(json.dumps(decision, indent=2, ensure_ascii=False, cls=NumpyEncoder))
    return decision

if __name__ == '__main__':
    run()