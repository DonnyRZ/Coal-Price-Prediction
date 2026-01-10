# 🔮 Coal AI Alpha: 基于 GRU-Attention 与 NLP 舆情的动力煤量化系统

> **Status:** 🚀 Production | **Model:** GRU + Attention | **Feature:** Multimodal (Price + NLP)

## 📖 项目简介 (Introduction)

本项目是一个针对**动力煤 (Coking Coal, JM)** 期货市场的全自动化量化预测系统。

区别于传统的纯技术面回测，本项目构建了一套**“AI + 基本面”**的混合驱动架构。系统后端利用 GitHub Actions 进行每日自动化推理，前端通过实时看板展示预测结果。

核心模型采用 **GRU (门控循环单元)** 结合 **Attention (注意力机制)**，并在输入端融合了基于 **NLP (自然语言处理)** 提取的政策面与市场情绪因子，旨在捕捉大宗商品市场中非线性的“量价”与“舆情”耦合关系。

---

## 🏗️ 系统数据流 (Data Flow)

本项目采用**云原生**的数据处理流，确保了从信息获取到决策输出的时效性。

```mermaid
graph TD
    subgraph "Data Sources (Heterogeneous)"
        A1[交易数据 (Price/Vol)]
        A2[CCTV 政策新闻]
        A3[机构研报 & 财经早餐]
    end

    subgraph "NLP Pipeline"
        B1[Jieba 分词 & 清洗]
        B2[金融情感词典匹配]
        B3[舆情因子生成 (Buzz/Risk)]
    end

    subgraph "Deep Learning Core"
        C1[时序对齐 & 归一化]
        C2[GRU-Attention 推理]
        C3[双轴动量策略生成]
    end

    A1 & A2 & A3 --> B1
    B1 --> B2 --> B3
    B3 --> C1 --> C2 --> C3
    C3 --> D[交互式看板 (Streamlit)]
```
# 🧠 核心建模过程 (Modeling Process)

## 1. 模型架构：GRU Ultimate
经过多轮迭代（Baseline -> LSTM -> GRU），最终模型选择了 GRU (Gated Recurrent Unit) 作为骨干网络。
* **计算效率**: 相比 LSTM，GRU 参数更少，收敛速度更快，且在期货市场 7-14 天这种中短周期的预测上，表现出了更好的抗过拟合能力。
* **注意力机制(Attention)** : 引入 Attention 层自动计算时间步权重（Alpha），使模型能够“关注”过去的重大转折点（如政策突发日），而非平均对待每一天的数据。
## 2. 网络拓扑 (Topology)
```Plaintext
[Input Layer] (5维特征: 价格动量 + 3维舆情)
      ⬇️
[GRU Layer 1] (Hidden=128, Dropout=0.1, Return_Seq=True)
      ⬇️
[GRU Layer 2] (Hidden=128, Return_Seq=True)
      ⬇️
[Attention Layer] (计算 Context Vector)
      ⬇️
[Output Layer] (Linear -> Log Return)
```
# 🗣️ NLP 特征工程：为何选择 Jieba？
在处理非结构化文本数据时，我们选择了传统的 Jieba 分词 + 领域词典 方案，而非目前英文界流行的 Transformer 类模型（如 FinBERT）。
## 1. 技术选型对比 (Trade-off)
| 特性     | Jieba + 领域词典 (本项目方案)                                    | FinBERT (预训练模型)                                                          |   |   |
|----------|------------------------------------------------------------------|-------------------------------------------------------------------------------|---|---|
| 语言适配 | 原生中文优化，完美适配“保供稳价”、“产能核增”等中国特色煤炭术语。 | 原生 FinBERT 为英文架构。中文版往往基于通用语料，对特定大宗商品黑话理解不足。 |   |   |
| 可解释性 | 极高。我们可以精确知道哪一个词（如“安检”）触发了风险因子飙升。   | 黑盒。难以归因具体的市场驱动力。                                              |   |   |
| 资源消耗 | 极低。CPU 毫秒级处理，适合轻量级云端环境。                       | 高。依赖 GPU 推理，部署成本高昂。                                             |   |   |
| 定制化   | 支持自定义词典，快速响应新政策术语。                             | 需要重新 Fine-tune，维护周期长。                                              |   |   |

## 2. 舆情因子构建
我们构建了 4 维情绪指纹，量化市场“软信息”：
* **Buzz (7d)**: 舆情热度。滚动 7 天的新闻提及频次（量在价先）。
* **Risk (MA7)**: 风险浓度。捕捉“监管、事故、双焦”等负面词汇。
* **Sentiment**: 情感得分。反映市场多空情绪倾向。
* **Vol Change**: 成交量变化。
  
# 📊 策略表现与动量修正 (Performance & Strategy)
### 1. 实战跟踪
模型在 2026 年初的单边行情中展现了优秀的趋势跟随能力。
### 2. 解决“滞后性”：双轴动量系统
针对时序模型在急涨急跌中容易出现的“保守滞后 (Lag)”问题，我们在策略层引入了动量确认机制：
* **价格轴 (Price Signal)**: 传统的均值回归信号（预测价 vs 当前价）。
* **动量轴 (Momentum Signal)**: 预测值的斜率变化 (Pred_Today - Pred_Yesterday)。
**策略逻辑**:当基准信号提示“看跌”（因价格涨太快偏离预测值），但动量轴显示红色高柱（模型正在剧烈上调目标价）时，系统自动判定为强趋势行情，发出“跟随趋势”信号，有效防止了牛市踏空。

# 📝 免责声明 (Disclaimer)
本系统仅用于量化策略研究与算法验证 (Research Purpose Only)。模型输出的任何信号均不构成投资建议。期货市场风险巨大，实盘交易请自行承担风险。
