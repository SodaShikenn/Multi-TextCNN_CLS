TRAIN_SAMPLE_PATH = './data/output/train_question_sample.txt'
TEST_SAMPLE_PATH = './data/output/test_question_sample.txt'
LABEL_PATH = './data/input/label.txt'

BERT_PAD_ID = 0
TEXT_LEN = 30

BERT_MODEL = 'bert-base-chinese'
MODEL_DIR = './data/output/models/'

BATCH_SIZE = 30

EMBEDDING_DIM = 768
NUM_FILTERS = 256
NUM_CLASSES = 12
FILTER_SIZES = [2, 3, 4]

EPOCH = 100
LR = 1e-3

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    print(torch.tensor([1,2,3]).to(DEVICE))