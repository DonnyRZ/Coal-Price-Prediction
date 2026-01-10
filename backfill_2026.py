import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import sys
import datetime
import math

# ----------------- 路径设置 -----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.gsheet_manager import read_from_sheet, write_to_sheet

# ================= 配置 =================
INPUT_TAB = "final_model_input"
OUTPUT_TAB = "prediction_results"
MODEL_PATH = "model_inference/best_gru_model.pth"
SCALER_PATH = "model_inference/scaler.pkl"

# 模型参数 (必须一致)
SEQ_LEN = 10
HIDDEN_SIZE = 128
INPUT_SIZE = 5

# ================= 1. 模型类定义 (复制粘贴) =================
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
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.1)
        self.attn = Attention(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.rnn(x)
        ctx = self.attn(out)
        return self.fc(ctx)

# ================= 2. 特征工程 (与 Predict 一致) =================
def engineer_features(df_raw):
    df = df_raw.copy()
    
    # 强制转换数值
    cols = ['open', 'high', 'low', 'close', 'volume', 'news_count', 'risk_score', 'sentiment_score']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', ascending=True, inplace=True)
    
    # 计算特征
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['vol_change'] = df['volume'].pct_change()
    df['buzz_7d'] = df['news_count'].rolling(window=7).sum()
    df['risk_ma7'] = df['risk_score'].rolling(window=7).mean()
    df['sent_ma7'] = df['sentiment_score'].rolling(window=7).mean()
    
    df.dropna(inplace=True)
    return df

# ================= 3. 补全逻辑 =================
def run_backfill():
    print("⏳ [Backfill] 开始补全 2026 年初至今的预测记录...")

    # 1. 读取数据
    df = read_from_sheet(INPUT_TAB)
    if df.empty:
        print("❌ 数据为空")
        return

    # 2. 特征工程
    df_feat = engineer_features(df)
    print(f"   数据预处理完成，可用行数: {len(df_feat)}")

    # 3. 加载模型 & Scaler
    if not os.path.exists(SCALER_PATH) or not os.path.exists(MODEL_PATH):
        print("❌ 模型或 Scaler 缺失")
        return

    scaler = joblib.load(SCALER_PATH)
    device = torch.device('cpu')
    model = Model_GRU_Attn(INPUT_SIZE, HIDDEN_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 4. 遍历 2026 年的每一天
    # 我们需要找到所有 2026 年的日期，然后对每一个日期进行“模拟预测”
    feature_cols = ['log_ret', 'vol_change', 'buzz_7d', 'risk_ma7', 'sent_ma7']
    
    results = []
    
    # 筛选出 2026 年以后的数据行作为“预测基准日”
    # 注意：我们需要保留之前的数据用于 lookback (SEQ_LEN)
    start_date = pd.to_datetime("2026-01-01")
    
    # 找到所有日期 >= 2026-01-01 的行索引
    target_indices = df_feat[df_feat['date'] >= start_date].index
    
    print(f"   需要补全的天数: {len(target_indices)}")

    for idx in target_indices:
        # 获取当前行 (T)
        row_T = df_feat.loc[idx]
        current_date = row_T['date']
        current_price = row_T['close']
        
        # 这里的 idx 是 DataFrame 的 label index。
        # 为了获取 sequence，我们需要 integer location (iloc)
        # 找到 idx 在 df_feat 中的位置
        integer_loc = df_feat.index.get_loc(idx)
        
        # 检查是否有足够的回顾窗口
        if integer_loc < SEQ_LEN:
            print(f"   ⚠️ 日期 {current_date.date()} 历史数据不足，跳过")
            continue

        # 构造输入序列 [T-SEQ_LEN : T] (包含T本身，作为最近的一天)
        # 注意：切片是 [start : end]，end 不包含，所以是 integer_loc + 1
        seq_data = df_feat.iloc[integer_loc - SEQ_LEN + 1 : integer_loc + 1]
        
        # 提取 5 个特征
        data_raw = seq_data[feature_cols].values
        
        # 归一化
        data_scaled = scaler.transform(data_raw)
        input_tensor = torch.from_numpy(data_scaled).float().unsqueeze(0)

        # 推理
        with torch.no_grad():
            pred_scaled = model(input_tensor).item()

        # 反归一化
        dummy = np.zeros((1, INPUT_SIZE))
        dummy[0, 0] = pred_scaled
        pred_log_ret = scaler.inverse_transform(dummy)[0, 0]

        # 计算目标价格 (T+1)
        predicted_price = current_price * math.exp(pred_log_ret)
        
        # 预测日期 (T+1)
        predict_date = current_date + datetime.timedelta(days=1)
        
        change_pct = (predicted_price - current_price) / current_price * 100
        signal = "🔴 看涨" if predicted_price > current_price else "🟢 看跌"

        print(f"   ✅ [补全] 基准: {current_date.date()} -> 预测: {predict_date.date()} | 价格: {predicted_price:.2f}")

        results.append({
            'predict_date': predict_date.strftime('%Y-%m-%d'),
            'base_date': current_date.strftime('%Y-%m-%d'),
            'predicted_price': round(predicted_price, 2),
            'current_price': round(current_price, 2),
            'change_pct': round(change_pct, 2),
            'signal': signal,
            'model_version': 'Backfill-2026', # 标记一下这是补全的数据
            'run_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    # 5. 写入 GSheet
    if results:
        df_res = pd.DataFrame(results)
        print("☁️ 正在写入 Google Sheets...")
        write_to_sheet(df_res, OUTPUT_TAB, mode='append')
        print("🎉 补全完成！")
    else:
        print("⚠️ 没有产生任何补全数据。")

if __name__ == "__main__":
    run_backfill()