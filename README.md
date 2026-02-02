# BERT-TextCNN: Multi-label Classification & Sentiment Analysis

A dual-task NLP system combining **BERT embeddings** with **TextCNN** for:

- **Multi-label Text Classification** - Assign multiple category labels to each input
- **Sentiment Analysis** - Classify sentiment as Positive / Neutral / Negative

> **This project integrates and improves upon my previous works:**
>
> - [TextCNN_CLS](https://github.com/SodaShikenn/TextCNN_CLS) - Single-label text classification
> - [LCF-ATEPC_ABSA](https://github.com/SodaShikenn/LCF-ATEPC_ABSA) - Aspect-Based Sentiment Analysis (ABSA)

## Dataset

The dataset consists of **e-commerce product Q&A data** from [Taobao](https://www.taobao.com/) and [JD.com](https://www.jd.com/) platforms.

**Classification Labels (12 categories):**

| ID | Label | Description |
| --- | --- | --- |
| 0 | 功效 | Efficacy |
| 1 | 适用人群 | Target Users |
| 2 | 使用方法 | Usage Method |
| 3 | 其他 | Other |
| 4 | 属性 | Attributes |
| 5 | 使用感受 | Usage Experience |
| 6 | 不良反应 | Side Effects |
| 7 | 竞品对比 | Competitor Comparison |
| 8 | 包装 | Packaging |
| 9 | 价格 | Price |
| 10 | 渠道 | Channel |
| 11 | 物流 | Logistics |

**Sentiment Labels (3 classes):**

| ID | Label | Description |
| --- | --- | --- |
| 0 | 中评 | Neutral |
| 1 | 好评 | Positive |
| 2 | 差评 | Negative |

## Features

| Module | Task | Output |
| ------ | ---- | ------ |
| `TextCNN_Single` | Single-label classification | 12 categories |
| `TextCNN_Multi` | Multi-label classification | Multiple labels per input |
| `TextCNN_Sentiment` | Sentiment analysis | Positive / Neutral / Negative |
| `online` | Production inference API | Batch prediction support |

## Quick Start

```bash
# Environment setup
conda activate nlp

# Training
cd TextCNN_Single && python train.py
cd TextCNN_Multi && python train.py
cd TextCNN_Sentiment && python train.py

# Inference
cd online && python demo.py
```

## API Usage

```python
# Classification (Multi-label)
from models.classification.functions import predict
label_ids, label_texts = predict(['保湿效果怎么样？', '小孩子可以用吗？'])

# Sentiment Analysis
from models.sentiment.functions import predict
label_ids, label_texts = predict([['是正品吗？', '效果不错']])
```

## Project Structure

```text
├── TextCNN_Single/      # Single-label classification training
├── TextCNN_Multi/       # Multi-label classification training
├── TextCNN_Sentiment/   # Sentiment analysis training
└── online/              # Production inference module
    └── models/
        ├── bert-base-chinese/   # Pre-trained BERT
        ├── classification/      # Classification inference
        └── sentiment/           # Sentiment inference
```

## Tech Stack

- **Model**: BERT (bert-base-chinese) + TextCNN
- **Framework**: PyTorch, HuggingFace Transformers
- **Environment**: Python 3.12, CUDA supported

---

[日本語版 README](README_ja.md)
