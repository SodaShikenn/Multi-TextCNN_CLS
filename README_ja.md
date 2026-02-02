# BERT-TextCNN: 複数ラベル分類 & 感情分析

**BERT埋め込み**と**TextCNN**を組み合わせたデュアルタスクNLPシステム:

- **複数ラベルテキスト分類** - 各入力に複数のカテゴリラベルを割り当て
- **感情分析** - ポジティブ / 中立 / ネガティブに分類

> **本プロジェクトは従来のプロジェクトを統合・改良:**
>
> - [TextCNN_CLS](https://github.com/SodaShikenn/TextCNN_CLS) - 単一ラベルテキスト分類
> - [LCF-ATEPC_ABSA](https://github.com/SodaShikenn/LCF-ATEPC_ABSA) - アスペクトベース感情分析 (ABSA)

## データセット

データセットは**[淘宝(Taobao)](https://www.taobao.com/)と[京東(JD.com)](https://www.jd.com/)のECサイト商品Q&Aデータ**で構成されています。

**分類ラベル（12カテゴリ）:**

| ID | ラベル | 説明 |
| --- | --- | --- |
| 0 | 功效 | 効能 |
| 1 | 适用人群 | 対象ユーザー |
| 2 | 使用方法 | 使用方法 |
| 3 | 其他 | その他 |
| 4 | 属性 | 属性 |
| 5 | 使用感受 | 使用感 |
| 6 | 不良反应 | 副作用 |
| 7 | 竞品对比 | 競合比較 |
| 8 | 包装 | パッケージ |
| 9 | 价格 | 価格 |
| 10 | 渠道 | 販売チャネル |
| 11 | 物流 | 物流 |

**感情ラベル（3クラス）:**

| ID | ラベル | 説明 |
| --- | --- | --- |
| 0 | 中评 | 中立 |
| 1 | 好评 | ポジティブ |
| 2 | 差评 | ネガティブ |

## 機能

| モジュール | タスク | 出力 |
| ---------- | ------ | ---- |
| `TextCNN_Single` | 単一ラベル分類 | 12カテゴリ |
| `TextCNN_Multi` | 複数ラベル分類 | 入力ごとに複数ラベル |
| `TextCNN_Sentiment` | 感情分析 | ポジティブ / 中立 / ネガティブ |
| `online` | 本番推論API | バッチ予測対応 |

## クイックスタート

```bash
# 環境設定
conda activate nlp

# 学習
cd TextCNN_Single && python train.py
cd TextCNN_Multi && python train.py
cd TextCNN_Sentiment && python train.py

# 推論
cd online && python demo.py
```

## API使用例

```python
# 分類（複数ラベル）
from models.classification.functions import predict
label_ids, label_texts = predict(['保湿効果はどうですか？', '子供でも使えますか？'])

# 感情分析
from models.sentiment.functions import predict
label_ids, label_texts = predict([['本物ですか？', '効果が良いです']])
```

## プロジェクト構成

```text
├── TextCNN_Single/      # 単一ラベル分類の学習
├── TextCNN_Multi/       # 複数ラベル分類の学習
├── TextCNN_Sentiment/   # 感情分析の学習
└── online/              # 本番推論モジュール
    └── models/
        ├── bert-base-chinese/   # 事前学習済みBERT
        ├── classification/      # 分類推論
        └── sentiment/           # 感情分析推論
```

## 技術スタック

- **モデル**: BERT (bert-base-chinese) + TextCNN
- **フレームワーク**: PyTorch, HuggingFace Transformers
- **環境**: Python 3.12, CUDA対応

---

[English README](README.md)
