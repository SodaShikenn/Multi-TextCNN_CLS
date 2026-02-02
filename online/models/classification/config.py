import os

BASE_PATH = os.path.dirname(__file__)

LABEL_PATH = os.path.join(BASE_PATH, './data/label.txt')

BERT_PAD_ID = 0
TEXT_LEN = 50

BERT_MODEL = os.path.join(BASE_PATH, '../bert-base-chinese/')
MODEL_DIR = os.path.join(BASE_PATH, './data/checkpoints/')


EMBEDDING_DIM = 768
NUM_FILTERS = 256
NUM_CLASSES = 12
FILTER_SIZES = [2, 3, 4]

CLS_WEIGHT_COEF = [0.5, 1.0]
CLS_PROB_BAR = 0.5
EPS = 1e-10

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
