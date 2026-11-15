# scripts/data/stock_selector.py
import os, json, warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
UNIVERSE = 'hs300'           # 股票池：'hs300'、'zz500' 或自定义列表
TOP_N = 5                    # 选股数量
DATA_DAYS = 120              # 历史天数
REBALANCE_FREQ = 'weekly'    # 调仓频率（暂未使用，仅记录）

# 数据存储路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'cn')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tushare Token（留空则自动使用 akshare）
TUSHARE_TOKEN = ''   # 填入你的真实 token，若为空则使用 akshare

# ==================== 数据获取（Tushare / AKShare 自动切换） ====================
if TUSHARE_TOKEN:
    import tushare as ts
    pro = ts.pro_api(TUSHARE_TOKEN)
    USE_TUSHARE = True
    print("✅ 使用 Tushare 数据源")
else:
    import akshare as ak
    USE_TUSHARE = False
    print("✅ 使用 AKShare 数据源")

def get_universe(name='hs300'):
    """获取股票代码列表（自动补全后缀）"""
    if name == 'hs300':
        if USE_TUSHARE:
            # 沪深300成分股 (Tushare)
            df = pro.index_weight(index_code='000300.SH', trade_date='20240101')  # 任取最近日期
            codes = df['con_code'].tolist()
        else:
            # AKShare
            df = ak.index_stock_cons(symbol="000300")
            codes = df['品种代码'].tolist()
    elif name == 'zz500':
        if USE_TUSHARE:
            df = pro.index_weight(index_code='000905.SH', trade_date='20240101')
            codes = df['con_code'].tolist()
        else:
            df = ak.index_stock_cons(symbol="000905")
            codes = df['品种代码'].tolist()
    else:
        codes = name if isinstance(name, list) else []
    # 统一添加后缀
    stocks = []
    for c in codes:
        c = str(c)
        if c.startswith('6'):
            stocks.append(c + '.SH')
        else:
            stocks.append(c + '.SZ')
    return stocks

def get_daily_data(stock, days):
    """获取单只股票的日线数据（前复权）"""
    if USE_TUSHARE:
        # Tushare: pro.daily (未复权) + adj_factor (复权因子)
        try:
            # 获取复权因子
            df_adj = pro.adj_factor(ts_code=stock, trade_date='')
            # 获取日线
            df = pro.daily(ts_code=stock, start_date=(datetime.now() - timedelta(days=days+30)).strftime('%Y%m%d'),
                           end_date=datetime.now().strftime('%Y%m%d'))
            if df.empty:
                return pd.DataFrame()
            df = df.merge(df_adj[['trade_date', 'adj_factor']], on='trade_date', how='left')
            # 前复权计算
            last_factor = df['adj_factor'].iloc[-1]
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col] * (df['adj_factor'] / last_factor)
            df.rename(columns={'open': '开盘', 'high': '最高', 'low': '最低',
                               'close': '收盘', 'vol': '成交量'}, inplace=True)
            return df.tail(days).reset_index(drop=True)
        except Exception as e:
            print(f"  {stock} Tushare获取失败: {e}")
            return pd.DataFrame()
    else:
        # AKShare 已内置前复权
        try:
            df = ak.stock_zh_a_hist(symbol=stock[:6], period="daily",
                                    start_date=(datetime.now() - timedelta(days=days+30)).strftime('%Y%m%d'),
                                    end_date=datetime.now().strftime('%Y%m%d'),
                                    adjust="qfq")
            return df.tail(days).reset_index(drop=True)
        except Exception as e:
            print(f"  {stock} AKShare获取失败: {e}")
            return pd.DataFrame()

# ==================== 因子计算 ====================
def compute_factors(stock_list, days=120):
    """计算因子原始值"""
    factor_data = []
    for code in stock_list:
        df = get_daily_data(code, days)
        if df is None or len(df) < 60:
            continue
        close = df['收盘'].values
        volume = df['成交量'].values
        # 均线
        ma20 = close[-20:].mean()
        ma60 = close[-60:].mean() if len(close) >= 60 else ma20
        trend = (ma20 / ma60 - 1) if ma60 != 0 else 0
        momentum = close[-1] / close[-20] - 1 if close[-20] != 0 else 0
        # 波动率
        returns = df['收盘'].pct_change().dropna()
        vol = returns.tail(20).std() if len(returns) >= 20 else 0
        # 换手率（用成交量替代，若有换手率数据可从 tushare 的 daily_basic 获取）
        turnover = volume[-20:].mean()
        vol_ratio = volume[-1] / volume[-20:].mean() if volume[-20:].mean() != 0 else 1
        factor_data.append({
            'code': code,
            'trend': trend,
            'momentum': momentum,
            'volatility': vol,
            'turnover': turnover,
            'vol_ratio': vol_ratio
        })
    return pd.DataFrame(factor_data)

def score_stocks(df):
    """因子标准化与综合打分"""
    df = df.copy()
    # 极值处理
    for col in ['trend', 'momentum', 'turnover', 'vol_ratio']:
        upper = df[col].quantile(0.95)
        lower = df[col].quantile(0.05)
        df[col] = df[col].clip(lower, upper)
    df['volatility'] = -df['volatility']  # 负向
    # 换手率过高惩罚
    med_turn = df['turnover'].median()
    df['turnover_adj'] = df['turnover'].apply(
        lambda x: x if x <= med_turn*2 else med_turn*2 + (x - med_turn*2)*0.3)
    df['turnover'] = df['turnover_adj']
    # 百分位排名
    for col in ['trend', 'momentum', 'volatility', 'turnover', 'vol_ratio']:
        df[col+'_rank'] = df[col].rank(pct=True)
    df['score'] = (df['trend_rank'] + df['momentum_rank'] + df['volatility_rank'] +
                   df['turnover_rank'] + df['vol_ratio_rank']) / 5
    return df.sort_values('score', ascending=False)

# ==================== 主流程 ====================
if __name__ == '__main__':
    print(f"开始选股，股票池：{UNIVERSE}，目标数量：{TOP_N}")
    stocks = get_universe(UNIVERSE)
    print(f"获取到 {len(stocks)} 只成分股")
    factor_df = compute_factors(stocks)
    if len(factor_df) == 0:
        print("无数据，退出")
        exit()
    scored_df = score_stocks(factor_df)
    selected = scored_df.head(TOP_N)['code'].tolist()
    print(f"本期选股结果：{selected}")
    # 保存结果
    result = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'universe': UNIVERSE,
        'selected': selected,
        'scores': scored_df[['code','score']].head(10).to_dict(orient='records')
    }
    with open(os.path.join(OUTPUT_DIR, 'selected_stocks.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到 {OUTPUT_DIR}/selected_stocks.json")