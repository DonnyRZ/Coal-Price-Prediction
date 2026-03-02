import pandas as pd
# 引入公共模块进行云端读写
# from gsheet_manager import read_from_sheet, write_to_sheet
import sys
import os

# ----------------- 路径修正开始 -----------------
# 获取当前脚本所在的目录 (data_pipeline)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (data_pipeline 的上一级)
project_root = os.path.dirname(current_dir)
# 将根目录加入 Python 搜索路径，这样才能找到 utils 包
sys.path.append(project_root)
# ----------------- 路径修正结束 -----------------

# 现在可以正常引用了
from utils.gsheet_manager import write_to_sheet, read_from_sheet
# 如果需要字典目录，也可以用这个 root 路径拼接
DICT_DIR = os.path.join(project_root, "sentiment_dicts")
# ================= 配置 =================
TAB_PRICES = "raw_prices"                  # 输入1：期货价格表
TAB_SENTIMENT = "daily_features_for_model" # 输入2：情绪打分表
TAB_FINAL = "final_model_input_v2"         # 输出：最终喂给模型的宽表 (v2 for safe rollout)


def _build_trading_calendar(df_price: pd.DataFrame) -> pd.DatetimeIndex:
    dates = pd.to_datetime(df_price["date"], errors="coerce").dropna().sort_values()
    return pd.DatetimeIndex(dates.unique())


def _map_to_next_trading_day(
    dates: pd.Series, calendar: pd.DatetimeIndex
) -> pd.Series:
    # Map each date to the next available trading day (same-day if already trading).
    # This avoids losing weekend/holiday sentiment while preventing look-ahead.
    cal_sorted = calendar.sort_values()
    mapped = pd.to_datetime(dates, errors="coerce")
    positions = cal_sorted.searchsorted(mapped, side="left")
    positions = positions.clip(0, len(cal_sorted) - 1)
    return pd.Series(cal_sorted[positions].values, index=dates.index)

def align_and_merge():
    print("⚖️ [Align] 启动：模型训练数据规整化...")
    
    # 1. 读取两份数据源
    df_price = read_from_sheet(TAB_PRICES)
    df_sent = read_from_sheet(TAB_SENTIMENT)
    
    # 检查数据是否读取成功
    if df_price.empty:
        print(f"❌ 错误：价格表 ({TAB_PRICES}) 为空，无法合并。")
        return
    if df_sent.empty:
        print(f"❌ 错误：情绪表 ({TAB_SENTIMENT}) 为空，无法合并。")
        return

    # 2. 数据预处理
    df_price["date"] = pd.to_datetime(df_price["date"], errors="coerce")
    df_sent["date"] = pd.to_datetime(df_sent["date"], errors="coerce")
    df_price = df_price.dropna(subset=["date"]).copy()
    df_sent = df_sent.dropna(subset=["date"]).copy()

    trading_calendar = _build_trading_calendar(df_price)
    if trading_calendar.empty:
        print("❌ 错误：无法构建交易日历 (价格日期为空)。")
        return

    # Map sentiment dates to the next trading day to avoid weekend gaps.
    df_sent["effective_date"] = _map_to_next_trading_day(
        df_sent["date"], trading_calendar
    )

    # Aggregate multiple news days that map to the same trading day.
    df_sent = (
        df_sent.groupby("effective_date", as_index=False)
        .agg(
            {
                "sentiment_score": "mean",
                "risk_score": "mean",
                "future_score": "mean",
                "conflict_score": "mean",
                "news_count": "sum",
            }
        )
        .rename(columns={"effective_date": "date"})
    )

    # 3. 执行合并 (Inner Join)
    # 逻辑：取交集。只保留“既有价格又有新闻情绪”的交易日。
    # 这样能自动剔除周末、节假日以及停盘日，防止空值进入模型。
    print(f"   正在合并：价格({len(df_price)}条) + 情绪({len(df_sent)}条)...")
    
    df_final = pd.merge(df_price, df_sent, on="date", how="left")
    # 🟢 填充缺失的新闻数据
    # 对于有价格但没新闻的“历史日子”，我们假设情绪是中性的 (0)
    fill_values = {
        'sentiment_score': 0.0,
        'risk_score': 0.0,
        'future_score': 0.0,
        'conflict_score': 0.0,
        'news_count': 0
    }
    df_final.fillna(value=fill_values, inplace=True)
    
    # 再次清洗：如果 Close 也是空的（理论上不会，因为是 left join price），丢弃
    df_final.dropna(subset=["close"], inplace=True)
    
    if df_final.empty:
        print("❌ 错误：合并后数据为空！请检查两张表的日期格式是否一致 (如 2026-01-08)。")
        return

    # 4. 强制规整列顺序 (Strict Column Ordering)
    # 这是最关键的一步，必须严格符合你训练模型时的输入特征顺序
    # 顺序：日期 -> 价格特征 -> 情绪特征 -> 统计特征
    target_cols = [
        'date', 
        'open', 'high', 'low', 'close', 'volume', 
        'sentiment_score', 
        'risk_score', 
        'future_score', 
        'conflict_score', 
        'news_count'  # 放在最后
    ]
    
    # 检查是否存在缺失列
    missing_cols = [c for c in target_cols if c not in df_final.columns]
    if missing_cols:
        print(f"⚠️ 警告：合并后缺失以下列，请检查源数据: {missing_cols}")
        # 如果缺列，可能导致后续写入或者模型报错，这里做个简单防守
        # 但通常如果 fetch 和 score 脚本正常，这里不会缺
    else:
        # 只保留目标列，并且强制排序
        df_final = df_final[target_cols]

    # 5. 按日期倒序排列 (最新的数据在第一行)
    df_final.sort_values("date", ascending=False, inplace=True)
    
    # 6. 写入云端
    print(f"   ✅ 合并完成，共生成 {len(df_final)} 条完整的训练样本。")
    print("   数据预览 (最新 3 条):")
    print(df_final.head(3))
    
    # 使用 overwrite 模式，保证 final 表永远是干净整洁的最新快照
    write_to_sheet(df_final, TAB_FINAL, mode="overwrite")

if __name__ == "__main__":
    align_and_merge()
