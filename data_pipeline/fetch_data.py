import pandas as pd
import akshare as ak
import feedparser
import requests
import datetime
import urllib.parse
import time
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
# from gsheet_manager import write_to_sheet, read_from_sheet

# ================= 配置中心 =================
RSS_QUERY = "焦煤 OR 焦炭 OR 动力煤 when:2d"
CCTV_FILTER_KEYWORDS = [
    "国务院", "发改委", "央行", "人民银行", "财政部", "统计局",
    "货币政策", "降准", "降息", "LPR", "社融", "信贷", "GDP", "CPI", "PPI",
    "经济工作", "扩大内需", "专项债", "稳增长",
    "能源", "煤炭", "煤矿", "保供", "产能", "安全生产", "安监", "碳达峰", "碳中和",
    "钢铁", "房地产", "基建", "住房", "建材", "高炉", "工信部"
]
PRICE_TAB_NAME = "raw_prices"

# ================= 辅助工具：核弹级去代理 =================
class TemporaryNoProxy:
    """
    上下文管理器：
    1. 临时移除系统代理变量
    2. 强制设置 NO_PROXY = "*" (这是解决顽固代理问题的关键)
    """
    def __enter__(self):
        # 1. 需要屏蔽的代理变量
        self.targets = [
            "http_proxy", "https_proxy", "ftp_proxy", "all_proxy",
            "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "ALL_PROXY"
        ]
        self.captured = {}
        
        # 备份并删除
        for key in self.targets:
            if key in os.environ:
                self.captured[key] = os.environ[key]
                del os.environ[key]
        
        # 2. 备份现有的 NO_PROXY (如果有)
        self.old_no_proxy = os.environ.get("NO_PROXY")
        self.old_no_proxy_caps = os.environ.get("no_proxy")
        
        # 3. 🟢 核心修改：强制设置 NO_PROXY 为通配符 "*"
        # 这告诉 requests 库：对所有域名都不使用代理
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 1. 恢复被删除的代理变量
        for key, value in self.captured.items():
            os.environ[key] = value
            
        # 2. 恢复旧的 NO_PROXY 设置
        if self.old_no_proxy is not None:
            os.environ["NO_PROXY"] = self.old_no_proxy
        else:
            # 如果原来没有，就删掉我们加的
            if "NO_PROXY" in os.environ: del os.environ["NO_PROXY"]
            
        if self.old_no_proxy_caps is not None:
            os.environ["no_proxy"] = self.old_no_proxy_caps
        else:
            if "no_proxy" in os.environ: del os.environ["no_proxy"]

# ================= 抓取逻辑 =================

def fetch_google_rss():
    print(f"📡 [Fetch] Google RSS ({RSS_QUERY})...")
    encoded_query = urllib.parse.quote(RSS_QUERY)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
    
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
        feed = feedparser.parse(resp.text)
        data = [{'date': e.published, 'title': e.title, 'link': e.link, 'source': 'Google_Industry'} for e in feed.entries]
        
        df = pd.DataFrame(data)
        write_to_sheet(df, "raw_google_rss", mode='append')
    except Exception as e:
        print(f"   ❌ RSS 抓取失败: {e}")

def fetch_cctv():
    print(f"📺 [Fetch] CCTV 新闻联播...")
    all_news = []
    # 使用增强版 NoProxy
    with TemporaryNoProxy():
        for i in range(3):
            target_date = datetime.datetime.now() - datetime.timedelta(days=i)
            date_str_api = target_date.strftime("%Y%m%d")
            date_str_std = target_date.strftime("%Y-%m-%d")
            try:
                news_df = ak.news_cctv(date=date_str_api)
                if not news_df.empty:
                    count = 0
                    for _, row in news_df.iterrows():
                        full = str(row['title']) + str(row['content'])
                        if any(k in full for k in CCTV_FILTER_KEYWORDS):
                            all_news.append({'date': date_str_std, 'title': row['title'], 'content': row['content'], 'source': 'CCTV_Macro'})
                            count += 1
                    print(f"   - {date_str_std}: {count} 条")
                time.sleep(1)
            except: pass
            
    df = pd.DataFrame(all_news)
    if not df.empty: write_to_sheet(df, "raw_cctv", mode='append')
    else: print("   ⚠️ 无有效宏观新闻")

def fetch_futures_price():
    print(f"💰 [Fetch] 正在抓取焦煤期货价格 (JM0)...")
    df_new = None
    
    # 动态计算日期范围
    today = datetime.datetime.now()
    start_date = (today - datetime.timedelta(days=200)).strftime("%Y%m%d")
    end_date = today.strftime("%Y%m%d")
    
    # 启用增强版 NoProxy
    with TemporaryNoProxy():
        try:
            # 尝试抓取
            df_slice = ak.futures_main_sina(
                symbol="JM0", 
                start_date=start_date, 
                end_date=end_date
            )
            
            if not df_slice.empty:
                rename_map = {
                    "日期": "date", "开盘价": "open", "最高价": "high",
                    "最低价": "low", "收盘价": "close", "成交量": "volume"
                }
                df_slice.rename(columns=rename_map, inplace=True)
                
                cols_to_keep = ['date', 'open', 'high', 'low', 'close', 'volume']
                available_cols = [c for c in cols_to_keep if c in df_slice.columns]
                
                df_new = df_slice[available_cols].copy()
                if 'date' in df_new.columns:
                    df_new['date'] = df_new['date'].astype(str)
                
                print(f"   ✅ 接口获取成功 ({start_date}-{end_date}): {len(df_new)} 行")
            else:
                print(f"   ⚠️ 接口返回空数据 ({start_date}-{end_date})")
                
        except Exception as e:
            print(f"   ❌ 价格抓取失败: {e}")

    # 代理已恢复，开始同步到 Google Sheets
    if df_new is not None and not df_new.empty:
        print("☁️ [GSheet] 同步历史价格进行合并...")
        df_history = read_from_sheet(PRICE_TAB_NAME)
        
        if not df_history.empty:
            if 'date' in df_history.columns:
                df_history['date'] = df_history['date'].astype(str)
            df_combined = pd.concat([df_history, df_new], ignore_index=True)
            df_combined.drop_duplicates(subset=['date'], keep='last', inplace=True)
            df_combined.sort_values('date', ascending=False, inplace=True)
        else:
            df_combined = df_new

        write_to_sheet(df_combined, PRICE_TAB_NAME, mode='overwrite')

if __name__ == "__main__":
    fetch_google_rss()
    fetch_cctv()
    fetch_futures_price()