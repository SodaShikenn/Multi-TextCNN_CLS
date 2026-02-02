import os

BASE_PATH = os.path.dirname(__file__)

BERT_PAD_ID = 0
QUESTION_TEXT_LEN = 50
ANSWER_TEXT_LEN = 50

BERT_MODEL = os.path.join(BASE_PATH, '../bert-base-chinese/')
MODEL_DIR = os.path.join(BASE_PATH, './data/checkpoints/')

EMBEDDING_DIM = 768
NUM_FILTERS = 256
NUM_CLASSES = 3
FILTER_SIZES = [2, 3, 4]

ID2LABEL = ['中评', '好评', '差评']

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Make a commit of emcapsulation of sentiment analysis and push to the github
