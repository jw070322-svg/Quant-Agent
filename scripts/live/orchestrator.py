# scripts/live/orchestrator.py
import os, json, subprocess
import pandas as pd
import numpy as np
import joblib
from openai import OpenAI

# ==================== 项目根目录 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
MODEL_CN_DIR = os.path.join(PROJECT_ROOT, 'models', 'cn')

# ==================== 配置 ====================
TICKER = '603986.SH'
MODEL_PATH = os.path.join(MODEL_CN_DIR, f'model_{TICKER}.pkl')
FEATURES_PATH = os.path.join(MODEL_CN_DIR, f'features_{TICKER}.txt')

# 如果没有模型，直接报错
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"请先训练 {TICKER} 的模型，并保存到 {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, 'r') as f:
    feature_columns = f.read().split(',')

# ==================== 获取最新上涨概率 ====================
# 这里用示例特征，实际需要从行情源获取
today_features = {
    'MA5': 150.5, 'MA10': 148.3, 'MA20': 145.1,
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
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# ==================== 定义工具：获取股票上涨概率 ====================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_probability",
            "description": f"获取 {TICKER} 未来1日上涨超过1%的概率。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码"}
                },
                "required": ["symbol"]
            }
        }
    }
]

def get_probability_impl(symbol: str):
    return json.dumps({
        "symbol": symbol,
        "up_probability": round(up_prob, 4),
        "note": "模型预测未来1日涨幅 > 1% 的概率"
    }, ensure_ascii=False)

# ==================== 调用 Claude Code（本地 CLI） ====================
def call_claude_code(prompt):
    try:
        result = subprocess.run(
            ['claude', '-p', prompt, '--output-format', 'text'],
            capture_output=True, text=True, timeout=120
        )
        return result.stdout.strip() if result.returncode == 0 else f"Claude Code 错误: {result.stderr}"
    except Exception as e:
        return f"调用 Claude Code 失败: {str(e)}"

# ==================== 协调器主逻辑 ====================
def orchestrate(user_query):
    # 第一步：让 DeepSeek 获取概率并初步分析
    messages = [{"role": "user", "content": user_query}]
    resp = client.chat.completions.create(model="deepseek-chat", messages=messages, tools=tools)
    msg = resp.choices[0].message
    messages.append(msg)

    if msg.tool_calls:
        for tc in msg.tool_calls:
            if tc.function.name == "get_stock_probability":
                args = json.loads(tc.function.arguments)
                result = get_probability_impl(args.get("symbol", TICKER))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        ds_response = client.chat.completions.create(model="deepseek-chat", messages=messages).choices[0].message.content
    else:
        ds_response = msg.content

    print("--- DeepSeek 分析 ---")
    print(ds_response)

    # 第二步：将 DeepSeek 的结果交给 Claude Code 深度分析
    claude_prompt = f"""你是一位资深A股量化交易顾问。基于以下技术分析结果，给出详细的操作建议和风险提示。

DeepSeek 技术分析结果：
{ds_response}

用户原始问题：{user_query}

请结合你的知识，补充：
1. 当前市场环境对该股的可能影响。
2. 策略可行性评估。
3. 潜在风险与对冲建议。
4. 最终交易决策（买入/卖出/观望）及理由。

用专业、清晰的中文回答。"""

    print("\n--- Claude Code 深度分析 ---")
    claude_response = call_claude_code(claude_prompt)
    print(claude_response)

    # 整合
    final = f"""## 📈 量化分析报告

### DeepSeek 快速预测：
{ds_response}

### Claude Code 深度决策：
{claude_response}

---
*以上分析由 AI 生成，不构成投资建议。*"""
    return final

if __name__ == "__main__":
    user_input = input("请输入你的问题（直接回车使用默认）: ")
    if not user_input:
        user_input = f"{TICKER} 今天上涨概率是多少？结合技术面给我一个操作建议。"
    report = orchestrate(user_input)
    print("\n" + "="*50)
    print(report)