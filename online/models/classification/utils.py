from .config import *

from transformers import logging
logging.set_verbosity_error()


def get_label():
    text = open(LABEL_PATH).read()
    id2label = text.split()
    return id2label, {v: k for k, v in enumerate(id2label)}


# 数字转序列函数
def ids2seq(ids):
    seq = [0] * NUM_CLASSES
    for i in ids:
        seq[int(i)] = 1
    return seq


# 序列转数字函数
def seq2ids(seq):
    return [k for k,v in enumerate(seq) if v==1]
