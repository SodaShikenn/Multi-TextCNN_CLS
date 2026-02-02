from torch.utils import data
from config import *
import torch
from transformers import BertTokenizer
from sklearn.metrics import classification_report
import pandas as pd

from transformers import logging
import numpy as np
logging.set_verbosity_error()

class Dataset(data.Dataset):
    def __init__(self, type='train'):
        super().__init__()
        if type == 'train':
            sample_path = TRAIN_SAMPLE_PATH
        elif type == 'test':
            sample_path = TEST_SAMPLE_PATH

        # self.lines = open(sample_path).readlines()
        self.lines = pd.read_csv(sample_path)
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        # text, label = self.lines[index].split('\t')
        text, labels = self.lines.loc[index, ['content', 'labels']].values
        tokened = self.tokenizer(text)
        input_ids = tokened['input_ids']
        mask = tokened['attention_mask']
        if len(input_ids) < TEXT_LEN:
            pad_len = (TEXT_LEN - len(input_ids))
            input_ids += [BERT_PAD_ID] * pad_len
            mask += [0] * pad_len
        target = ids2seq(labels.split('|'))
        return torch.tensor(input_ids[:TEXT_LEN]), torch.tensor(mask[:TEXT_LEN]), torch.tensor(target)


def get_label():
    text = open(LABEL_PATH).read()
    id2label = text.split()
    return id2label, {v: k for k, v in enumerate(id2label)}


# Convert IDs to multi-hot sequence
def ids2seq(ids):
    seq = [0] * NUM_CLASSES
    for i in ids:
        seq[int(i)] = 1
    return seq


# Convert multi-hot sequence to IDs
def seq2ids(seq):
    return [k for k,v in enumerate(seq) if v==1]


def evaluate(pred, true):
    pred = np.array(pred)
    pred = np.where(pred > CLS_PROB_BAR, 1, 0)

    correct_num, predict_num, gold_num = 0, 0, 0

    for pred_, true_ in zip(pred, true):
        predict_num += 1
        gold_num += 1
        if seq2ids(pred_) == seq2ids(true_):
            correct_num += 1

    precision = correct_num / (predict_num + EPS)
    recall = correct_num / (gold_num + EPS)
    f1_score = 2 * precision * recall / (precision + recall + EPS)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


if __name__ == '__main__':
    dataset = Dataset()
    loader = data.DataLoader(dataset, batch_size=2)
    print(next(iter(loader)))
    exit()

    print(ids2seq('0|6'.split('|')))
    print(seq2ids([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
