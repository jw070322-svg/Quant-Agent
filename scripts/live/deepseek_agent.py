# scripts/live/deepseek_agent.py
import os, json
import pandas as pd
import numpy as np
import joblib
from openai import OpenAI

# ==================== 项目根目录定位 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
MODEL_US_DIR = os.path.join(PROJECT_ROOT, 'models', 'us')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# ==================== 配置：使用 AAPL 模型作为示例 ====================
TICKER = 'AAPL'  # 你可以改为 'AMZN', 'GOOGL' 等
MODEL_PATH = os.path.join(MODEL_US_DIR, f'model_{TICKER}.pkl')
FEATURES_PATH = os.path.join(MODEL_US_DIR, f'features_{TICKER}.txt')

# 如果模型不存在，给出明确提示
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型文件 {MODEL_PATH} 不存在，请先训练 {TICKER} 的模型。")

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, 'r') as f:
    feature_columns = f.read().split(',')

# ==================== 今日最新特征（示例） ====================
# 实际使用时，应从行情源获取最新数据并计算特征，这里沿用之前的示例
today_features = {
    'MA5': 260.5, 'MA10': 258.3, 'MA20': 255.1,
    'MACD': 2.1, 'MACD_hist': 0.3,
    'K': 70.2, 'D': 65.8, 'J': 79.0,
    'RSI': 62.5,
    'BB_width': 8.5, 'BB_position': 0.65,
    'volatility_5': 0.015, 'volatility_20': 0.018,
    'close_vs_MA20': 0.02, 'volume_ratio': 1.1,
    'roc_7': 0.03, 'roc_14': 0.05
}

X_today = pd.DataFrame([today_features])[feature_columns]
up_prob = model.predict_proba(X_today)[0, 1]

# ==================== 初始化 DeepSeek 客户端 ====================
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("未设置环境变量 DEEPSEEK_API_KEY")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_probability",
            "description": f"获取 {TICKER} 未来上涨超过 1% 的概率。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码，如 AAPL"}
                },
                "required": ["symbol"]
            }
        }
    }
]

def get_probability_impl(symbol: str):
    result = {
        "symbol": symbol,
        "up_probability": round(up_prob, 4),
        "note": "模型预测未来2日涨幅 > 1% 的概率"
    }
    return json.dumps(result, ensure_ascii=False)

def ask_deepseek(user_question: str):
    messages = [{"role": "user", "content": user_question}]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools
    )
    msg = response.choices[0].message
    messages.append(msg)

    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            if tool_call.function.name == "get_stock_probability":
                args = json.loads(tool_call.function.arguments)
                function_result = get_probability_impl(args.get("symbol", TICKER))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": function_result
                })
        final_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        return final_response.choices[0].message.content
    else:
        return msg.content

# ==================== 示例提问 ====================
if __name__ == "__main__":
    question = f"{TICKER} 今天上涨概率是多少？结合技术面给我一个操作建议。"
    answer = ask_deepseek(question)
    print("🤖 DeepSeek 回答：")
    print(answer)