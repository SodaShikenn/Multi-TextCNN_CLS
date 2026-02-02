# Bert TextCNN Classification

## Project Structure

```
code/
├── TextCNN_Single/              # Single-label classification
│   └── data/
│       ├── input/               # Input data (label.txt)
│       └── output/
│           ├── models/          # Trained models (model_weights_single_*.pth)
│           ├── train_question_sample.txt
│           └── test_question_sample.txt
├── TextCNN_Multi/               # Multi-label classification
│   └── data/
│       ├── input/               # Input data (label.txt)
│       └── output/
│           ├── models/          # Trained models (model_weights_multi_*.pth)
│           ├── train_question_sample.txt
│           └── test_question_sample.txt
└── online/                      # Online inference
    ├── demo.py
    ├── test.py
    ├── test_question_sample.csv
    └── models/
        ├── bert-base-chinese/   # Local BERT model
        └── classification/
            ├── config.py
            ├── model.py
            ├── utils.py
            ├── functions.py
            └── data/
                ├── label.txt
                └── checkpoints/  # Model checkpoints
```

## Configuration

### TextCNN_Single (Single-label Classification)

| Variable | Path |
|----------|------|
| TRAIN_SAMPLE_PATH | `./data/output/train_question_sample.txt` |
| TEST_SAMPLE_PATH | `./data/output/test_question_sample.txt` |
| LABEL_PATH | `./data/input/label.txt` |
| BERT_MODEL | `bert-base-chinese` (HuggingFace) |
| MODEL_DIR | `./data/output/models/` |

### TextCNN_Multi (Multi-label Classification)

| Variable | Path |
|----------|------|
| TRAIN_SAMPLE_PATH | `./data/output/train_question_sample.txt` |
| TEST_SAMPLE_PATH | `./data/output/test_question_sample.txt` |
| LABEL_PATH | `./data/input/label.txt` |
| BERT_MODEL | `bert-base-chinese` (HuggingFace) |
| MODEL_DIR | `./data/output/models/` |

### online/models/classification (Online Inference)

| Variable | Path |
|----------|------|
| LABEL_PATH | `./data/label.txt` |
| BERT_MODEL | `../bert-base-chinese/` (Local) |
| MODEL_DIR | `./data/checkpoints/` |

## Model Naming Convention

- Single-label models: `model_weights_single_{epoch}.pth`
- Multi-label models: `model_weights_multi_{epoch}.pth`

## Usage

### Training

```bash
# Single-label
cd TextCNN_Single
python train.py

# Multi-label
cd TextCNN_Multi
python train.py
```

### Testing

```bash
# Single-label
cd TextCNN_Single
python test.py

# Multi-label
cd TextCNN_Multi
python test.py
```

### Prediction

```bash
# Single-label
cd TextCNN_Single
python predict.py

# Multi-label
cd TextCNN_Multi
python predict.py
```

## Environment

- Python 3.12
- PyTorch
- Transformers (HuggingFace)
- Conda environment: `nlp`

```bash
conda activate nlp
```
