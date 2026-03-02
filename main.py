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
    page_title="Coal AI | Coal Futures Dashboard",
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
        if not client:
            return pd.DataFrame(), "GCP credentials not found. Check GCP_SERVICE_ACCOUNT_JSON."

        SHEET_NAME = "Coal_Data_Master"
        TAB_NAME = "prediction_results"

        try:
            sh = client.open(SHEET_NAME)
            ws = sh.worksheet(TAB_NAME)
            raw_data = ws.get_all_values()
        except Exception as e:
            return pd.DataFrame(), f"Failed to open sheet/tab: {e}"

        if len(raw_data) < 2:
            return pd.DataFrame(), "prediction_results is empty."

        headers = [h.strip() for h in raw_data[0]]
        rows = raw_data[1:]
        df = pd.DataFrame(rows, columns=headers)

        if "predict_date" not in df.columns:
            return pd.DataFrame(), "prediction_results is missing 'predict_date' column."

        # Type conversion
        df["predict_date"] = pd.to_datetime(df["predict_date"])

        cols_num = ["current_price", "predicted_price", "change_pct"]
        for c in cols_num:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Momentum
        df = df.sort_values("predict_date", ascending=True)
        df["pred_slope"] = df["predicted_price"].diff()

        # Return latest first
        df = df.sort_values("predict_date", ascending=False)
        return df, ""

    except Exception as e:
        return pd.DataFrame(), f"Unexpected error: {e}"

# ================= 3. 侧边栏 =================
with st.sidebar:
    st.markdown("[⬅️ Back to main site (nice-ai.dev)](https://nice-ai.dev)")

    st.title("🔮 Coal AI Alpha")
    st.markdown("---")

    st.subheader("🤖 Model Settings")
    st.info("**Architecture**: GRU + Attention")
    st.caption("Lookback: 10 Days | Data: 2023-2025")

    st.markdown("---")

    st.subheader("📖 Chart Guide")
    st.markdown("""
    **Dual-axis overlay:**
    1. **Lines (left axis)**: Price trend
       - 🔘 Gray: Actual price
       - 🔴 Red: AI target
    2. **Bars (right axis)**: Momentum
       - 📊 Taller bars mean stronger momentum.
       - **Red**: Bullish momentum
       - **Green**: Bearish momentum
    """)
    st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')}")

# ================= 4. 主界面 =================
st.title("Coal Futures (JM) Intelligent Strategy Dashboard")

df, load_error = load_data_direct()

if not df.empty:
    latest = df.iloc[0]
    prev = df.iloc[1] if len(df) > 1 else latest
    slope_val = latest['pred_slope']
    
    # --- KPI ---
    cols = st.columns(5)
    with cols[0]:
        st.metric("Target Date", latest['predict_date'].strftime('%Y-%m-%d'))
    with cols[1]:
        st.metric("Base Price", f"{latest['current_price']:.0f}")
    with cols[2]:
        delta_pred = latest['predicted_price'] - prev['predicted_price']
        st.metric("AI Target", f"{latest['predicted_price']:.1f}", delta=f"{delta_pred:+.1f}")
    with cols[3]:
        st.metric("Momentum", f"{slope_val:+.1f}", delta=None)
    with cols[4]:
        base_signal = str(latest['signal'])
        if "看涨" in base_signal:
            display_signal = "Bullish (Long)"
            color = "red"
        elif "看跌" in base_signal:
            display_signal = "Bearish (Short)"
            color = "green"
        else:
            display_signal = base_signal
            color = "blue"
        st.markdown("**Final Signal**")
        st.markdown(f":{color}[**{display_signal}**]")

    st.markdown("---")

    # --- 核心图表区 ---
    tab1, tab2 = st.tabs(["📉 Trend & Momentum (Overlay)", "📊 Backtest"])

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
            st.info("Not enough data yet to display the trend chart.")

    with tab2:
        img_path = "assets/model_performance_report.png"
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.warning("Backtest image not found.")

else:
    if load_error:
        st.info(f"☁️ No data available. {load_error}")
    else:
        st.info("☁️ No data available. Make sure Google Sheets access is configured and prediction_results has data.")

# ================= 5. 项目文档 (README) =================
st.markdown("---")
# [新增] GitHub 仓库链接 (使用 Shields.io 徽章风格)
# 🚨 请将下方的链接替换为你真实的 GitHub 仓库地址
GITHUB_REPO_URL = "https://github.com/dachou5224/coal-price-prediction"

st.markdown(
    f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <a href="{GITHUB_REPO_URL}" target="_blank" style="text-decoration: none;">
            <img src="https://img.shields.io/badge/GitHub-View_Source_Code-181717?style=for-the-badge&logo=github" alt="GitHub Repo">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
# 使用折叠面板，默认收起，保持页面整洁
with st.expander("📖 About Model", expanded=False):
    readme_path = os.path.join(current_dir, "front_page_readme.md")
    
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        
        # 渲染 Markdown
        st.markdown(readme_content)
        
    except FileNotFoundError:
        st.warning("⚠️ front_page_readme.md not found in the project root.")
    except Exception as e:
        st.error(f"无法加载文档: {str(e)}")

# 页脚签名
st.markdown(
    """
    <div style='text-align: center; color: grey; font-size: 0.8em; margin-top: 30px;'>
        Coal AI Alpha System | Powered by GRU-Attention & NLP | Deployed via Docker & GitHub Actions
    </div>
    """,
    unsafe_allow_html=True
)
