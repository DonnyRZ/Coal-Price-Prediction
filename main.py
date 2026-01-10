import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
from datetime import datetime

# ----------------- 路径设置 -----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.gsheet_manager import get_client

# ================= 1. 页面配置 =================
st.set_page_config(
    page_title="Coal AI | 焦煤策略看板",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 风格设置
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 配色方案
COLOR_UP = '#d62728'     # 红
COLOR_DOWN = '#2ca02c'   # 绿
COLOR_ACTUAL = '#333333' # 深灰
COLOR_BAR_ALPHA = 0.25   # 柱状图透明度

# ================= 2. 数据加载 =================
@st.cache_data(ttl=0) 
def load_data_direct():
    try:
        client = get_client()
        if not client: return pd.DataFrame()

        SHEET_NAME = "Coal_Data_Master" 
        TAB_NAME = "prediction_results"
        
        try:
            sh = client.open(SHEET_NAME)
            ws = sh.worksheet(TAB_NAME)
            raw_data = ws.get_all_values()
        except Exception:
            return pd.DataFrame()

        if len(raw_data) < 2: return pd.DataFrame()

        headers = [h.strip() for h in raw_data[0]]
        rows = raw_data[1:]
        df = pd.DataFrame(rows, columns=headers)

        if 'predict_date' not in df.columns: return pd.DataFrame()

        # 类型转换
        df['predict_date'] = pd.to_datetime(df['predict_date'])
        
        cols_num = ['current_price', 'predicted_price', 'change_pct']
        for c in cols_num:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # 计算动量
        df = df.sort_values('predict_date', ascending=True)
        df['pred_slope'] = df['predicted_price'].diff()
        
        # 倒序返回
        df = df.sort_values('predict_date', ascending=False)
        return df

    except Exception:
        return pd.DataFrame()

# ================= 3. 侧边栏 =================
with st.sidebar:
    st.title("🔮 Coal AI Alpha")
    st.markdown("---")
    
    st.subheader("🤖 模型参数")
    st.info("**架构**: GRU + Attention")
    st.caption("Lookback: 10 Days | Data: 2023-2025")
    
    st.markdown("---")
    
    st.subheader("📖 图表解读")
    st.markdown("""
    **双轴叠加图：**
    1. **折线 (左轴)**: 价格走势
       - 🔘 灰色: 真实价格
       - 🔴 红色: AI 目标价
    2. **柱状 (右轴)**: 趋势动量
       - 📊 柱子越高，模型追涨/杀跌的力度越大。
       - **红柱**: 动量向上 (Bullish)
       - **绿柱**: 动量向下 (Bearish)
    """)
    st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')}")

# ================= 4. 主界面 =================
st.title("焦煤期货 (JM) 智能策略看板")

df = load_data_direct()

if not df.empty:
    latest = df.iloc[0]
    prev = df.iloc[1] if len(df) > 1 else latest
    slope_val = latest['pred_slope']
    
    # --- KPI ---
    cols = st.columns(5)
    with cols[0]:
        st.metric("目标日期", latest['predict_date'].strftime('%Y-%m-%d'))
    with cols[1]:
        st.metric("基准价", f"{latest['current_price']:.0f}")
    with cols[2]:
        delta_pred = latest['predicted_price'] - prev['predicted_price']
        st.metric("AI 目标价", f"{latest['predicted_price']:.1f}", delta=f"{delta_pred:+.1f}")
    with cols[3]:
        st.metric("趋势动量", f"{slope_val:+.1f}", delta=None)
    with cols[4]:
        base_signal = str(latest['signal'])
        color = "red" if "看涨" in base_signal else "green"
        st.markdown("**最终信号**")
        st.markdown(f":{color}[**{base_signal}**]")

    st.markdown("---")

    # --- 核心图表区 ---
    tab1, tab2 = st.tabs(["📉 趋势与动量 (叠加)", "📊 历史回测"])

    with tab1:
        if len(df) > 1:
            plot_df = df.sort_values('predict_date')
            dates = plot_df['predict_date']
            
            # 创建画布
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # --- 层级 1: 柱状图 (右轴) ---
            ax2 = ax1.twinx()
            slope_colors = [COLOR_UP if s >= 0 else COLOR_DOWN for s in plot_df['pred_slope']]
            ax2.bar(dates, plot_df['pred_slope'], color=slope_colors, alpha=COLOR_BAR_ALPHA, width=0.6, label='Momentum', zorder=1)
            
            # 优化右轴范围
            max_slope = max(abs(plot_df['pred_slope'].max()), abs(plot_df['pred_slope'].min()), 1.0)
            ax2.set_ylim(-max_slope * 3, max_slope * 3)
            ax2.set_ylabel("Momentum", color='gray', fontsize=9)
            ax2.tick_params(axis='y', labelcolor='gray')
            ax2.grid(False)
            
            # --- 层级 2: 折线图 (左轴) ---
            ax1.plot(dates, plot_df['current_price'], label='Actual Price', color=COLOR_ACTUAL, linewidth=2.5, marker='o', markersize=5, zorder=10)
            ax1.plot(dates, plot_df['predicted_price'], label='AI Target', color=COLOR_UP, linestyle='--', linewidth=2, marker='x', zorder=10)
            
            ax1.set_ylabel("Price (CNY)", fontweight='bold')
            ax1.set_title("Price Trend vs AI Momentum", fontsize=14, fontweight='bold', loc='left')
            ax1.grid(True, alpha=0.3)
            
            # --- 图例合并 ---
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            from matplotlib.patches import Patch
            patch_up = Patch(color=COLOR_UP, alpha=COLOR_BAR_ALPHA, label='Pos Momentum')
            patch_down = Patch(color=COLOR_DOWN, alpha=COLOR_BAR_ALPHA, label='Neg Momentum')
            ax1.legend(lines_1 + [patch_up, patch_down], labels_1 + ['Up Trend', 'Down Trend'], loc='upper left')
            
            # 🟢 修复重点：设置日期刻度定位器 (Locator)
            # 强制按“天”显示，解决 01-06 出现两次的问题
            if len(dates) < 30:
                ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            else:
                ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
                
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            
            # 渲染
            st.pyplot(fig)
            
        else:
            st.info("数据积累中，暂无趋势图。")

    with tab2:
        img_path = "assets/model_performance_report.png"
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.warning("未找到回测报告图片")

else:
    st.info("☁️ 暂无数据")