# 📈 焦煤期货 AI 预测系统 (Coal Futures Prediction Backend)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-GRU-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research_Preview-yellow)

基于 **GRU + Attention** 深度学习模型与 **NLP 情感分析** 的期货价格预测系统。本项目实现了从数据抓取、清洗、特征对齐到模型推理的全流程自动化。

---

## ⚠️ 免责声明 (Disclaimer)

**重要提示：本项目仅供计算机科学、金融工程及人工智能领域的学术研究与学习使用。**

1.  **非投资建议**：本项目产生的所有预测结果、图表及信号仅作为算法验证的参考，**不构成任何形式的投资建议或交易信号**。
2.  **风险提示**：期货市场具有极高的波动性和风险。依据本项目模型结果进行的任何真实交易，后果由使用者自行承担。
3.  **数据来源**：项目使用公开网络数据（RSS, CCTV, 财经接口），不保证数据的准确性、及时性或完整性。

**Do NOT trade based on this repo. This is a research project.**

---

## 💡 项目架构 (Architecture)

本项目采用 **"后端大统一 (Monorepo)"** 架构，利用 GitHub Actions 实现每日定时自动化运行。系统通过 Google Sheets 作为轻量级云端数据库。

### 数据链路 (Data Pipeline)
1.  **Fetch (`data_pipeline/fetch_data.py`)**:
    * 抓取 Google RSS 新闻（行业动态）。
    * 抓取 CCTV 新闻联播（宏观政策，含智能关键词筛选）。
    * 抓取新浪期货/AKShare 价格数据（支持去代理直连）。
2.  **Merge (`data_pipeline/merge_process.py`)**:
    * 清洗非结构化文本，标准化日期格式。
    * 基于文本内容去重，生成统一的新闻池。
3.  **Score (`data_pipeline/score_sentiment.py`)**:
    * 基于行业专用词典（Sentiment Dictionary）进行 NLP 分析。
    * 计算多维特征：`Sentiment` (情绪), `Risk` (风险), `Future` (预期), `Conflict` (分歧)。
4.  **Align (`data_pipeline/align_model_data.py`)**:
    * **时空对齐**：使用 Left Join 将交易数据与情绪数据对齐。
    * **冷启动处理**：自动填充缺失的历史情绪数据，确保模型输入完整。

### 模型推理 (Inference)
* **模型架构**: GRU (Gated Recurrent Unit) + Attention Mechanism。
* **输入特征**: 10 维特征（包含价格序列、成交量变化、多维 NLP 情绪指数）。
* **预测目标**: T+1 日的对数收益率及涨跌信号。
* **部署**: PyTorch 模型 (`.pth`) 配合 `RobustScaler` 进行实时归一化推理。

---

## 📂 目录结构

```text
futures-prediction-backend/
├── .github/workflows/      # GitHub Actions 自动化配置
├── data_pipeline/          # 数据工程 (ETL) 脚本
│   └── sentiment_dicts/    # NLP 情感词典
├── model_inference/        # 模型推理核心
│   ├── predict.py          # 推理脚本
│   ├── best_gru_model.pth  # 训练好的 PyTorch 模型
│   └── scaler.pkl          # 对应的归一化参数
├── utils/                  # 公共工具箱 (GSheet 连接器)
├── requirements.txt        # Python 依赖
└── README.md               # 项目说明

```

# 🚀 快速开始 (Quick Start)

### 1. 环境准备
确保本地已安装 Python 3.9+ 环境。

### 2. 克隆仓库

### 3. 安装依赖

### 4. 配置密钥 (Local Debug)
本项目依赖 Google Sheets API。

在 Google Cloud Platform 申请 Service Account。

下载 JSON 密钥文件，重命名为 service_account_key.json。

将该文件放入项目根目录（注意：该文件已在 .gitignore 中，切勿上传到 GitHub）。

### 5. 运行脚本
你可以单独运行某个模块，建议在根目录下执行：


```bash
# 1. 抓取数据
python data_pipeline/fetch_data.py

# 2. 清洗合并
python data_pipeline/merge_process.py

# 3. 情感打分
python data_pipeline/score_sentiment.py

# 4. 数据对齐
python data_pipeline/align_model_data.py

# 5. 模型推理
python model_inference/predict.py
```

# ☁️ 部署 (Deployment)
本项目配置了 GitHub Actions，实现每日自动化运行。

Fork 本仓库。

在仓库 Settings -> Secrets and variables -> Actions 中添加 Secret：

Name: GCP_SERVICE_ACCOUNT_JSON

Value: (复制 service_account_key.json 的全部内容)

Workflow 将在每日 UTC 14:30 (北京时间 22:30) 自动触发。

# 🛠 技术栈
语言: Python 3.9

深度学习: PyTorch

数据源: AkShare, Feedparser

NLP: Jieba (中文分词)

CI/CD: GitHub Actions

Database: Google Sheets (via gspread)

📜 License
MIT License. See LICENSE file for details.
