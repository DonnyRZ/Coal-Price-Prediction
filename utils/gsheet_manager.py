import os
import json
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ================= 🚨 本地调试专用：强制代理 =================
# 根据你的实际情况取消注释
if not os.environ.get("GITHUB_ACTIONS"):
    # os.environ["http_proxy"] = "http://127.0.0.1:7890"
    # os.environ["https_proxy"] = "http://127.0.0.1:7890"
    pass
# ========================================================

SHEET_NAME = "Coal_Data_Master"

def get_client():
    """获取 GSheet 客户端连接"""
    json_str = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    
    if json_str:
        creds_dict = json.loads(json_str)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            creds_dict, 
            ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        )
        return gspread.authorize(creds)
    # 2. 回退：本地文件 (Local Debug)
    # 获取 utils 文件夹的上一级（即项目根目录）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接出 key 的绝对路径
    key_path = os.path.join(base_dir, "..", "service_account_key.json")
    
    # 检查是否存在
    if os.path.exists(key_path):
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            key_path, # 使用绝对路径
            ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        )
        return gspread.authorize(creds)
    elif os.path.exists("service_account_key.json"):
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            "service_account_key.json", 
            ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        )
        return gspread.authorize(creds)
        
    else:
        print("❌ 错误：未找到 GCP 凭据 (Env or Local File)")
        return None

def write_to_sheet(df, tab_name, mode='append'):
    """通用写入函数 (更稳健的表头检查)"""
    if df.empty:
        print(f"⚠️ [{tab_name}] 数据为空，跳过写入。")
        return

    print(f"☁️ [GSheet] 正在写入: {tab_name} (模式: {mode})...")
    client = get_client()
    if not client: return

    try:
        sheet = client.open(SHEET_NAME)
        # 尝试获取子表，不存在则创建
        try:
            worksheet = sheet.worksheet(tab_name)
        except gspread.WorksheetNotFound:
            print(f"      + 创建新子表: {tab_name}")
            worksheet = sheet.add_worksheet(title=tab_name, rows=1000, cols=20)
        
        # 1. 强制转字符串 (防止 JSON 错误)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_timedelta64_dtype(df[col]):
                df[col] = df[col].astype(str)

        df = df.fillna('')
        data_to_write = df.values.tolist()

        if mode == 'append':
            # 🟢 修复核心：使用 row_values(1) 检查第一行是否有内容
            header_row = worksheet.row_values(1)
            
            if not header_row:
                print("      (检测到空表，正在初始化表头...)")
                # 写表头 + 数据
                worksheet.update([df.columns.values.tolist()] + data_to_write)
            else:
                # 检查表头是否匹配 (可选警告)
                if header_row != df.columns.values.tolist():
                    # 这里只打印警告，不强制中断，防止频繁报错
                    # print(f"      ⚠️ 警告: 云端表头 {header_row} 与本地 {df.columns.values.tolist()} 不完全一致")
                    pass
                
                # 直接追加数据
                worksheet.append_rows(data_to_write)
            
            print(f"   ✅ 已成功写入 {len(df)} 行数据")
            
        elif mode == 'overwrite':
            worksheet.clear()
            worksheet.update([df.columns.values.tolist()] + data_to_write)
            print(f"   ✅ 已覆盖更新，当前共 {len(df)} 行")

    except Exception as e:
        print(f"❌ 写入失败: {e}")

def read_from_sheet(tab_name):
    """从 Google Sheets 读取数据并转为 DataFrame"""
    print(f"📥 [GSheet] 正在读取: {tab_name}...")
    client = get_client()
    if not client: return pd.DataFrame()

    try:
        sheet = client.open(SHEET_NAME)
        worksheet = sheet.worksheet(tab_name)
        
        # 🟢 改用 get_all_values 以便调试 (它返回列表的列表，而不是字典)
        all_rows = worksheet.get_all_values()
        
        if not all_rows:
            print(f"   ⚠️ {tab_name} 为空")
            return pd.DataFrame()
            
        # 假设第一行是表头
        headers = all_rows[0]
        data = all_rows[1:]
        
        if not headers:
             print(f"   ⚠️ {tab_name} 第一行看起来是空的")
             return pd.DataFrame()

        df = pd.DataFrame(data, columns=headers)
        
        # 🐛 调试打印：如果没找到 date 列，打印一下当前的列名，帮你定位问题
        if 'date' not in df.columns and 'Date' not in df.columns:
            print(f"   ❌ 严重错误：未在 {tab_name} 中找到 'date' 列！")
            print(f"      当前云端实际列名(表头)是: {df.columns.tolist()}")
            print(f"      (这通常意味着第一行是数据而不是表头，请清空 Sheet 重试)")
            return pd.DataFrame() # 返回空以免后续报错
            
        print(f"   ✅ 读取成功: {len(df)} 行")
        return df
        
    except gspread.WorksheetNotFound:
        print(f"❌ 读取失败: 子表 '{tab_name}' 不存在")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ 读取失败 ({tab_name}): {e}")
        return pd.DataFrame()