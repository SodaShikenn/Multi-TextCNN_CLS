
TRAIN_SAMPLE_PATH = './data/input/tb_question_answer_sentiment_7k.xlsx'
TEST_SAMPLE_PATH = './data/input/jd_question_answer_sentiment_2k.xlsx'

BERT_PAD_ID = 0
QUESTION_TEXT_LEN = 50
ANSWER_TEXT_LEN = 50

BERT_MODEL = '../huggingface/bert-base-chinese/'
MODEL_DIR = './data/output/models/'

BATCH_SIZE = 100

EMBEDDING_DIM = 768
NUM_FILTERS = 256
NUM_CLASSES = 3
FILTER_SIZES = [2, 3, 4]

ID2LABEL = ['中评', '好评', '差评']

EPOCH = 100
LR = 1e-3

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    print(torch.tensor([1,2,3]).to(DEVICE))