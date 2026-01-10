import pandas as pd
import jieba
import datetime
# 引入公共模块进行云端读写
# from gsheet_manager import read_from_sheet, write_to_sheet, get_client, SHEET_NAME
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
from utils.gsheet_manager import write_to_sheet, read_from_sheet, get_client, SHEET_NAME
# 如果需要字典目录，也可以用这个 root 路径拼接
DICT_DIR = os.path.join(project_root, "sentiment_dicts")
# ================= 配置 =================
INPUT_TAB_NAME = "news_merged_ready"          # 输入：清洗整合后的新闻数据
OUTPUT_TAB_NAME = "daily_features_for_model"  # 输出：每日多维特征
# DICT_DIR = "sentiment_dicts"                  # 字典文件夹路径

# ================= 核心类：多维情感分析器 =================
class FinancialSentimentAnalyzer:
    def __init__(self, dict_path):
        self.dict_path = dict_path
        # 初始化所有维度的字典
        self.dicts = {
            'pos': set(),
            'neg': set(),
            'risk': set(),
            'future': set(),
            'conflict': set()
        }
        self.stopwords = set()
        
        # 1. 加载本地 txt 字典
        self.load_dictionaries()
        
        # 2. 注入行业专用词 (硬编码保底)
        self.augment_coal_vocabulary()

    def load_dictionaries(self):
        print(f"📚 [Init] 正在从 {self.dict_path} 加载多维字典...")
        
        # 映射文件名 (请确保 sentiment_dicts 文件夹里有这些文件)
        file_map = {
            'pos': 'pos_words.txt',
            'neg': 'neg_words.txt',
            'risk': 'risk_keywords.txt',
            'future': 'forward_keywords.txt',
            'conflict': '逆接成分表.txt',
            'stop': 'cn_stopwords.txt'
        }

        if not os.path.exists(self.dict_path):
            print(f"   ⚠️ 警告：字典文件夹 {self.dict_path} 不存在，将仅使用硬编码增强词。")
            return

        for key, filename in file_map.items():
            file_path = os.path.join(self.dict_path, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        words = set(line.strip() for line in f if line.strip())
                        if key == 'stop':
                            self.stopwords.update(words)
                        else:
                            self.dicts[key].update(words)
                except Exception as e:
                    print(f"      ❌ 读取 {filename} 失败: {e}")
            else:
                print(f"      ⚠️ 警告: 未找到字典 {filename}，{key} 维度分数可能为0")

    def augment_coal_vocabulary(self):
        """注入煤炭/黑色系期货专用的行业黑话"""
        # 利多/利空
        coal_pos = {"上涨", "大涨", "长协", "保供", "补库", "去库", "旺季", "紧平衡", "支撑", "回升", "复苏", "利好", "紧缺", "强劲", "拉升", "涨停", "看多", "做多"}
        coal_neg = {"下跌", "大跌", "跳水", "累库", "积压", "满库", "淡季", "宽松", "疲软", "低迷", "回落", "下行", "跌停", "绿盘", "看空", "做空", "滞销", "观望"}
        
        # 风险 (Risk): 监管、安全、不可抗力
        coal_risk = {"监管", "约谈", "限价", "打压", "事故", "矿难", "安检", "环保", "停产", "检修", "限产", "整改", "通报", "违规", "罚款"}
        
        # 预期 (Future): 展望未来
        coal_future = {"预计", "展望", "预测", "将来", "有望", "预期", "计划", "目标", "将会"}

        self.dicts['pos'].update(coal_pos)
        self.dicts['neg'].update(coal_neg)
        self.dicts['risk'].update(coal_risk)
        self.dicts['future'].update(coal_future)
        
        print(f"   ✅ 已注入行业增强词: Pos({len(coal_pos)}), Neg({len(coal_neg)}), Risk({len(coal_risk)})")

    def analyze(self, text):
        """
        计算单条文本的【多维】情感得分
        返回: [sentiment_score, risk_score, future_score, conflict_score]
        """
        if not isinstance(text, str):
            return pd.Series([0.0, 0.0, 0.0, 0.0])
            
        # 1. 分词
        words = jieba.lcut(text)
        # 过滤停用词和单字
        words = [w for w in words if w not in self.stopwords and len(w) > 1]
        
        if not words:
            return pd.Series([0.0, 0.0, 0.0, 0.0])
            
        total_words = len(words)
        
        # 2. 统计各维度命中数
        counts = {'pos': 0, 'neg': 0, 'risk': 0, 'future': 0, 'conflict': 0}
        
        for w in words:
            if w in self.dicts['pos']: counts['pos'] += 1
            elif w in self.dicts['neg']: counts['neg'] += 1
            
            # 注意：同一个词可能既是 Pos 也是 Future（视字典定义），所以这里不用 elif，而是独立判断
            if w in self.dicts['risk']: counts['risk'] += 1
            if w in self.dicts['future']: counts['future'] += 1
            if w in self.dicts['conflict']: counts['conflict'] += 1
        
        # 3. 计算指标
        
        # A. 情感分 (Sentiment): (-1, 1)
        # 逻辑：(正面 - 负面) / 总情感词数
        # 注意：分母加1防止除零
        sentiment_denom = counts['pos'] + counts['neg'] + 1
        sentiment_score = (counts['pos'] - counts['neg']) / sentiment_denom
        
        # B. 风险关注度 (Risk): (0, 1) -> 占总词数的比例
        risk_score = counts['risk'] / total_words
        
        # C. 预期强度 (Future): (0, 1)
        future_score = counts['future'] / total_words
        
        # D. 分歧/转折度 (Conflict): (0, 1)
        conflict_score = counts['conflict'] / total_words
        
        return pd.Series([sentiment_score, risk_score, future_score, conflict_score])

# ================= 主程序 =================
def generate_daily_features():
    print("\n🧠 [Score] 启动每日多维特征生成...")

    # 1. 从云端读取
    df = read_from_sheet(INPUT_TAB_NAME)
    
    if df.empty:
        print("❌ 错误：未从云端获取到新闻数据，无法计算。")
        return

    # 2. 初始化分析器
    analyzer = FinancialSentimentAnalyzer(DICT_DIR)

    # 3. 计算每条新闻的得分
    print(f"   正在计算 {len(df)} 条新闻的多维得分...")
    df['full_text'] = df['full_text'].fillna('').astype(str)
    
    # apply 返回一个 DataFrame (4列)，直接赋值给原来的 df
    # 列名顺序对应 analyze 返回的 Series 顺序
    score_cols = ['sentiment_raw', 'risk_raw', 'future_raw', 'conflict_raw']
    df[score_cols] = df['full_text'].apply(analyzer.analyze)
    
    # 4. 按日期聚合 (Resample 到日频)
    df['date'] = pd.to_datetime(df['date'])
    df['date_str'] = df['date'].dt.date.astype(str)
    
    # 聚合逻辑：所有分数都取【均值】，同时保留新闻数量
    daily_features = df.groupby('date_str').agg({
        'sentiment_raw': 'mean',
        'risk_raw': 'mean',
        'future_raw': 'mean',
        'conflict_raw': 'mean',
        'source': 'count'
    }).reset_index()
    
    # 重命名列以对齐模型输入
    daily_features.rename(columns={
        'date_str': 'date', 
        'sentiment_raw': 'sentiment_score',
        'risk_raw': 'risk_score',
        'future_raw': 'future_score',
        'conflict_raw': 'conflict_score',
        'source': 'news_count'
    }, inplace=True)
    # 🟢 核心修改：调整列顺序，确保 news_count 在最后
    cols_order = ['date', 'sentiment_score', 'risk_score', 'future_score', 'conflict_score', 'news_count']
    daily_features = daily_features[cols_order]
    
    print(f"   ✅ 今日计算完成，共生成 {len(daily_features)} 个交易日的数据。")

    # 5. 智能合并：读取云端历史 -> 合并 -> 去重 -> 覆盖
    print("☁️ [GSheet] 正在同步历史数据以进行合并...")
    client = get_client()
    if not client: return

    try:
        sheet = client.open(SHEET_NAME)
        try:
            worksheet = sheet.worksheet(OUTPUT_TAB_NAME)
            existing_records = worksheet.get_all_records()
            df_history = pd.DataFrame(existing_records)
        except:
            df_history = pd.DataFrame()
            print("      (历史数据表不存在，将新建)")

        if not df_history.empty:
            # 统一日期格式
            if 'date' in df_history.columns:
                df_history['date'] = df_history['date'].astype(str)
            
            # 确保历史数据里也有这些列（防止历史数据是旧版，缺少列）
            for col in ['risk_score', 'future_score', 'conflict_score']:
                if col not in df_history.columns:
                    df_history[col] = 0.0 # 缺失补0
            
            # 合并
            df_combined = pd.concat([df_history, daily_features], ignore_index=True)
            # 去重 (保留最新)
            df_combined.drop_duplicates(subset=['date'], keep='last', inplace=True)
            # 排序
            df_combined.sort_values('date', inplace=True)
        else:
            df_combined = daily_features

        # 6. 全量覆盖回写
        write_to_sheet(df_combined, OUTPUT_TAB_NAME, mode='overwrite')
        
        print("\n🎉 [Success] 每日多维情绪指数已更新！")
        print("   数据预览 (最后 3 天):")
        print(df_combined.tail(3))
        
    except Exception as e:
        print(f"❌ 指数合并/写入失败: {e}")

if __name__ == "__main__":
    generate_daily_features()