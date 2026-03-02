import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
import sys
import datetime
import math

# ----------------- 路径修正 -----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
# -------------------------------------------

from utils.gsheet_manager import read_from_sheet, write_to_sheet

# ================= 配置 =================
INPUT_TAB = "final_model_input_v2"    # 包含 Open, Close, Sentiment, News_Count 等 (v2)
OUTPUT_TAB = "prediction_results_v2"
MODEL_PATH = os.path.join(current_dir, "best_gru_model.pth")
SCALER_PATH = os.path.join(current_dir, "scaler.pkl")

# 🚨 必须与训练参数严格一致
SEQ_LEN = 10         # 回顾过去 10 天
HIDDEN_SIZE = 128    # 隐藏层大小
DROPOUT = 0.1        # (推理时不起作用，但类定义需要)
INPUT_SIZE = 5       # 特征数量 (log_ret, vol_change, buzz_7d, risk_ma7, sent_ma7)

# ================= 1. 模型类定义 (必须完全一致) =================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        energy = torch.tanh(self.attn(hidden_states))
        attention = self.v(energy).squeeze(2)
        alpha = torch.softmax(attention, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), hidden_states)
        return context.squeeze(1)

class Model_GRU_Attn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True, dropout=DROPOUT)
        self.attn = Attention(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64), 
            nn.ReLU(), 
            nn.Dropout(DROPOUT), 
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.rnn(x)
        ctx = self.attn(out)
        return self.fc(ctx)

# ================= 2. 实时特征工程 =================
def engineer_features(df_raw):
    """
    将原始数据 (Price, Sentiment) 转换为模型需要的 5 个特征:
    ['log_ret', 'vol_change', 'buzz_7d', 'risk_ma7', 'sent_ma7']
    """
    df = df_raw.copy()
    
    # 确保按时间正序 (旧->新) 计算指标
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', ascending=True, inplace=True)
    
    # 1. log_ret (对数收益率)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. vol_change (成交量变化率)
    df['vol_change'] = df['volume'].pct_change()
    
    # 3. buzz_7d (7天新闻热度总和) -> 对应 news_count
    df['buzz_7d'] = df['news_count'].rolling(window=7).sum()
    
    # 4. risk_ma7 (7天风险均值) -> 对应 risk_score
    df['risk_ma7'] = df['risk_score'].rolling(window=7).mean()
    
    # 5. sent_ma7 (7天情绪均值) -> 对应 sentiment_score
    df['sent_ma7'] = df['sentiment_score'].rolling(window=7).mean()
    
    # 移除因计算 rolling 产生的空值 (前7行)
    df.dropna(inplace=True)
    
    # 仅保留模型需要的列，且顺序必须对
    feature_cols = ['log_ret', 'vol_change', 'buzz_7d', 'risk_ma7', 'sent_ma7']
    
    # 返回: 特征矩阵, 最后一行收盘价(用于还原), 最后一行日期
    return df[feature_cols].values, df['close'].iloc[-1], df['date'].iloc[-1]

# ================= 3. 推理主逻辑 =================
def run_prediction():
    print("🔮 [Predict] 启动 PyTorch (GRU+Attention) 推理...")

    # 1. 检查文件
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"❌ 错误：模型文件或 Scaler 缺失。\n   路径: {MODEL_PATH}")
        return

    # 2. 读取原始数据
    df = read_from_sheet(INPUT_TAB)
    if df.empty:
        print("❌ 云端数据为空")
        return
    # 🟢 修复1：强制类型转换 (String -> Float)
    # 定义所有必须是数字的列
    numeric_cols = [
        'open', 'high', 'low', 'close', 'volume', 
        'sentiment_score', 'risk_score', 'future_score', 'conflict_score', 'news_count'
    ]
    
    # 循环转换，无法转换的变成 NaN
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 去除任何包含空值的行 (比如转换失败的 dirty data)
    df.dropna(subset=numeric_cols, inplace=True)

    print(f"   ✅ 数据加载并清洗完成: {len(df)} 行")
    # 3. 特征工程
    print("   正在计算衍生特征 (log_ret, MA7)...")
    try:
        # data_values 是只有 5 列的 numpy 数组
        data_values, current_price, current_date = engineer_features(df)
    except Exception as e:
        print(f"❌ 特征工程失败: {e}")
        print("   可能原因：数据量不足 7 天，无法计算 MA7 指标。")
        return

    # 4. 归一化 (Scaler)
    print("   正在加载 Scaler 并归一化...")
    try:
        scaler = joblib.load(SCALER_PATH)
        data_scaled = scaler.transform(data_values)
    except Exception as e:
        print(f"❌ 归一化失败: {e}")
        return

    # 5. 构造时间序列输入
    if len(data_scaled) < SEQ_LEN:
        print(f"❌ 有效数据不足 {SEQ_LEN} 天 (扣除MA7计算损耗)，无法预测")
        return

    # 取最后 SEQ_LEN 行作为输入
    input_seq = data_scaled[-SEQ_LEN:] 
    # 转 Tensor: [1, 10, 5]
    input_tensor = torch.from_numpy(input_seq).float().unsqueeze(0)

    # 6. 加载模型
    print("   正在加载冠军模型...")
    device = torch.device('cpu')
    model = Model_GRU_Attn(INPUT_SIZE, HIDDEN_SIZE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 7. 预测
    print("   正在推理...")
    with torch.no_grad():
        # 模型输出的是：归一化后的 log_ret (因为训练目标是 column 0)
        pred_scaled_log_ret = model(input_tensor).item()

    # 8. 反归一化 (还原真实涨幅)
    # Scaler 期望 5 列，我们只预测了第 1 列 (log_ret)
    # 构造 dummy 数组: [pred, 0, 0, 0, 0]
    dummy_row = np.zeros((1, INPUT_SIZE))
    dummy_row[0, 0] = pred_scaled_log_ret
    
    real_log_ret = scaler.inverse_transform(dummy_row)[0, 0]
    
    # 9. 计算目标价格
    # 逻辑: P_next = P_current * exp(log_ret)
    predicted_price = current_price * math.exp(real_log_ret)
    
    change_pct = (predicted_price - current_price) / current_price * 100
    signal = "🔴 看涨 (做多)" if predicted_price > current_price else "🟢 看跌 (做空)"
    
    # 格式化日期
    current_date_str = current_date.strftime('%Y-%m-%d')
    predict_date_str = (current_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"\n🔮 [Result] 预测报告:")
    print(f"   📅 基准日期: {current_date_str} (收盘: {current_price:.2f})")
    print(f"   🚀 预测日期: {predict_date_str} (T+1)")
    print(f"   🌊 预测对数收益: {real_log_ret:.6f}")
    print(f"   💰 目标价格: {predicted_price:.2f}")
    print(f"   📈 预期涨幅: {change_pct:.2f}%  [{signal}]")

    # 10. 写入结果
    result_df = pd.DataFrame([{
        'predict_date': predict_date_str,
        'base_date': current_date_str,
        'predicted_price': round(predicted_price, 2),
        'current_price': round(current_price, 2),
        'change_pct': round(change_pct, 2),
        'signal': signal,
        'model_version': 'GRU-Attn-v1',
        'run_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }])
    
    write_to_sheet(result_df, OUTPUT_TAB, mode='append')

if __name__ == "__main__":
    run_prediction()
