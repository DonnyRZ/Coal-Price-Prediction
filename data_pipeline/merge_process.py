import pandas as pd
import datetime
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
# 输入表 (Raw Data)
TAB_RSS = "raw_google_rss"
TAB_CCTV = "raw_cctv"

# 输出表 (Merged Data)
TAB_OUTPUT = "news_merged_ready"

def clean_and_merge():
    print("🔄 [Merge] 启动：从云端读取原始数据并标准化...")
    
    # 1. 从 Google Sheets 读取原始数据
    df_rss = read_from_sheet(TAB_RSS)
    df_cctv = read_from_sheet(TAB_CCTV)

    processed_dfs = []
    
    # ================= 处理 RSS 数据 =================
    # RSS 结构: date, title, link, source
    # 目标: 构造 full_text = title * 3
    if not df_rss.empty:
        print(f"   正在处理 RSS 数据 ({len(df_rss)} 条)...")
        try:
            # 日期标准化：处理带时区的 ISO 格式
            # errors='coerce' 会把无法解析的变成 NaT (Not a Time)，方便后续过滤
            df_rss['date'] = pd.to_datetime(df_rss['date'], utc=True, errors='coerce')
            # 转为北京时间并去掉时区信息
            df_rss['date'] = df_rss['date'].dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
            
            # 构造 full_text (标题重复3次以增加权重)
            # fillna('') 防止标题为空导致报错
            df_rss['full_text'] = (df_rss['title'].fillna('').astype(str) + " ") * 3
            
            # 提取需要的列
            processed_dfs.append(df_rss[['date', 'full_text', 'source']])
        except Exception as e:
            print(f"   ⚠️ 处理 RSS 数据出错: {e}")

    # ================= 处理 CCTV 数据 =================
    # CCTV 结构: date, title, content, source
    # 目标: 构造 full_text = title + content
    if not df_cctv.empty:
        print(f"   正在处理 CCTV 数据 ({len(df_cctv)} 条)...")
        try:
            # 日期标准化：CCTV 已经是 YYYY-MM-DD 字符串，但为了保险再次转换
            df_cctv['date'] = pd.to_datetime(df_cctv['date'], errors='coerce')
            
            # 设定时间为当天 19:00:00 (新闻联播时间)
            # 注意：只有非空日期才能加时间，否则会报错
            mask = df_cctv['date'].notna()
            df_cctv.loc[mask, 'date'] = df_cctv.loc[mask, 'date'] + datetime.timedelta(hours=19)
            
            # 构造 full_text (标题 + 正文)
            df_cctv['full_text'] = df_cctv['title'].fillna('').astype(str) + " " + df_cctv['content'].fillna('').astype(str)
            
            # 提取需要的列
            processed_dfs.append(df_cctv[['date', 'full_text', 'source']])
        except Exception as e:
            print(f"   ⚠️ 处理 CCTV 数据出错: {e}")

    # ================= 合并与输出 =================
    if processed_dfs:
        # 上下拼接
        df_final = pd.concat(processed_dfs, ignore_index=True)
        
        # 1. 再次清洗日期：删除所有日期解析失败 (NaT) 的行
        # 这一步非常关键，能过滤掉所有脏数据
        df_final = df_final.dropna(subset=['date'])
        
        # 2. 去重
        # 逻辑：如果 full_text 完全一样，视为重复抓取
        # keep='first' 保留第一次出现的
        df_final.drop_duplicates(subset=['full_text'], keep='first', inplace=True)
        
        # 3. 排序 (按时间倒序，最新的在前面)
        df_final.sort_values('date', ascending=False, inplace=True)
        
        print(f"   合并完成，共 {len(df_final)} 条有效数据。")
        
        # 4. 写入云端 (使用覆盖模式 overwrite)
        # 为什么用覆盖？因为 Merged 表是中间态，每次根据最新的 Raw 重新生成一遍是最干净的。
        # 如果 Raw 表数据量非常大(几万条)，可以考虑 append，但目前阶段 overwrite 更稳健，不会有重复数据困扰。
        write_to_sheet(df_final, TAB_OUTPUT, mode='overwrite')
        
    else:
        print("   ⚠️ 无有效数据可合并，跳过写入。")

if __name__ == "__main__":
    clean_and_merge()