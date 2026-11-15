# scripts/data/download_us_data.py
import os, time, random
import yfinance as yf
import pandas as pd

# HANDOVER FIX (Phase 0): this file lives directly in `data/`, so ascending two
# levels escaped the project. The resolved output used to be
# `F:\quant agent\data\us`, outside quant_project. Now it stays in-tree.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_US_DIR = os.path.join(PROJECT_ROOT, 'data', 'us')
os.makedirs(DATA_US_DIR, exist_ok=True)

TICKERS = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA', 'META', 'NFLX']
START_DATE = '2019-01-01'
END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d')

def download_and_save(ticker):
    print(f"Downloading {ticker} ...")
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if df.empty:
            print(f"  ⚠️ {ticker} 数据为空，跳过")
            return

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        save_path = os.path.join(DATA_US_DIR, f"{ticker}.csv")
        df.to_csv(save_path, index=False)
        print(f"  ✅ 已保存 {save_path}，共 {len(df)} 行")
    except Exception as e:
        print(f"  ❌ 下载 {ticker} 失败: {e}")

if __name__ == "__main__":
    for ticker in TICKERS:
        download_and_save(ticker)
        # 随机休眠2~5秒，避免触发限流
        delay = random.uniform(2, 5)
        print(f"  ⏳ 等待 {delay:.1f} 秒...")
        time.sleep(delay)
    print("\n✅ 所有美股数据下载完成。")