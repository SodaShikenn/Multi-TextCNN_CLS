from torch.utils import data
from config import *
import torch
from transformers import BertTokenizer
from sklearn.metrics import classification_report
import pandas as pd

from transformers import logging
logging.set_verbosity_error()

class Dataset(data.Dataset):
    def __init__(self, type='train'):
        super().__init__()
        if type == 'train':
            sample_path = TRAIN_SAMPLE_PATH
        elif type == 'test':
            sample_path = TEST_SAMPLE_PATH

        df = pd.read_excel(TRAIN_SAMPLE_PATH)
        df.dropna(subset=['questionContent', 'answerContent', 'sentiment'], how='any', inplace=True)
        self.lines = df.reset_index()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        question_content, answer_content, sentiment = \
            self.lines.loc[index, ['questionContent', 'answerContent', 'sentiment']].values
        # 问题文本编码
        question_tokened = self.tokenizer(question_content)
        question_input_ids = question_tokened['input_ids']
        question_mask = question_tokened['attention_mask']
        if len(question_input_ids) < QUESTION_TEXT_LEN:
            pad_len = (QUESTION_TEXT_LEN - len(question_input_ids))
            question_input_ids += [BERT_PAD_ID] * pad_len
            question_mask += [0] * pad_len
        else:
            question_input_ids = question_input_ids[:QUESTION_TEXT_LEN]
            question_mask = question_mask[:QUESTION_TEXT_LEN]
        # 答案文本编码
        answer_tokened = self.tokenizer(answer_content)
        answer_input_ids = answer_tokened['input_ids']
        answer_mask = answer_tokened['attention_mask']
        if len(answer_input_ids) < QUESTION_TEXT_LEN:
            pad_len = (QUESTION_TEXT_LEN - len(answer_input_ids))
            answer_input_ids += [BERT_PAD_ID] * pad_len
            answer_mask += [0] * pad_len
        else:
            answer_input_ids = answer_input_ids[:QUESTION_TEXT_LEN]
            answer_mask = answer_mask[:QUESTION_TEXT_LEN]
        # 问题+答案，作为整体传给模型
        input_ids = question_input_ids + answer_input_ids
        mask = question_mask + answer_mask
        # 目标值（0-中评，1好评，2-差评）
        target = 2 if sentiment == -1 else sentiment
        return torch.tensor(input_ids), torch.tensor(mask), torch.tensor(target)


def evaluate(pred, true, target_names=None, output_dict=False):
    return classification_report(
        true,
        pred,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0,
    )


if __name__ == '__main__':
    dataset = Dataset()
    loader = data.DataLoader(dataset, batch_size=2)
    print(next(iter(loader)))


