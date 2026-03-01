# 🔮 Coal AI Alpha: GRU‑Attention + NLP Sentiment for Coal Futures

> **Status:** 🚀 Production | **Model:** GRU + Attention | **Features:** Multimodal (Price + NLP)

## 📖 Overview

This project is an automated quantitative prediction system for **coking coal futures (JM)**.
Instead of pure technical backtesting, it uses a hybrid **“AI + fundamentals”** approach: daily
data is processed in the backend and results are shown on a live dashboard.

The core model is **GRU** with **Attention**, and the inputs include **policy/news sentiment**
extracted via lightweight NLP to capture non‑linear interactions between price and narrative.

---

## 🏗️ Data Flow

```mermaid
graph TD
    subgraph "Data Sources"
        A1[Price/Volume]
        A2[CCTV Policy News]
        A3[Research Reports & Market News]
    end

    subgraph "NLP Pipeline"
        B1[Jieba Tokenization & Cleaning]
        B2[Domain Sentiment Dictionaries]
        B3[Sentiment Factors (Buzz/Risk)]
    end

    subgraph "Deep Learning Core"
        C1[Time Alignment & Scaling]
        C2[GRU-Attention Inference]
        C3[Dual-Axis Momentum Signal]
    end

    A1 & A2 & A3 --> B1
    B1 --> B2 --> B3
    B3 --> C1 --> C2 --> C3
    C3 --> D[Streamlit Dashboard]
```

---

## 🧠 Modeling

### 1) Architecture
After iterations (Baseline → LSTM → GRU), the system uses **GRU**:
* **Efficiency:** fewer parameters than LSTM, faster convergence.
* **Attention:** highlights key time steps (e.g., policy shocks) instead of uniform weighting.

### 2) Topology
```Plaintext
[Input Layer] (5 features: price momentum + 3 sentiment dims)
      ⬇️
[GRU Layer 1] (Hidden=128, Dropout=0.1, Return_Seq=True)
      ⬇️
[GRU Layer 2] (Hidden=128, Return_Seq=True)
      ⬇️
[Attention Layer]
      ⬇️
[Output Layer] (Linear -> Log Return)
```

---

## 🗣️ NLP Features

We use **Jieba + domain dictionaries** instead of heavy Transformer models:

* **Language fit:** optimized for Chinese energy policy vocabulary.
* **Explainability:** you can trace which term drives a signal.
* **Lightweight:** CPU‑friendly for daily runs.

### Sentiment Factors
* **Buzz (7d):** rolling news volume.
* **Risk (MA7):** intensity of risk keywords.
* **Sentiment:** polarity score.
* **Vol Change:** volume momentum.

---

## 📊 Strategy Logic

To reduce lag in fast trends, the system uses a dual‑axis signal:
* **Price axis:** mean‑reversion vs. target price.
* **Momentum axis:** slope of predicted price (Pred_Today − Pred_Yesterday).

If base signal is bearish but momentum is strongly rising, the system follows the trend.

---

## 📝 Disclaimer

This system is for research and educational purposes only.  
Outputs are **not** investment advice. Futures trading carries high risk.
