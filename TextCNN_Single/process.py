import pandas as pd
from config import *
from utils import *

def label_to_id(string):
   _, label2id = get_label()
   label_texts = string.split('/')
   label_ids = [label2id[text] for text in label_texts]
   return '|'.join([str(id) for id in label_ids])

def trans_label(input_path, output_path):
    df = pd.read_excel(input_path)
    df['labels'] = df['category'].apply(label_to_id)
    df.to_csv(output_path, index=None)

def count_text_len():
   text_len = []
   df = pd.read_csv(TRAIN_SAMPLE_PATH)
   for content in df['content']:
       text_len.append(len(content))
   print(len([i for i in text_len if i>50])) # 1
   print(len(text_len)) # def count_text_len():
   text_len = []
   df = pd.read_csv(TRAIN_SAMPLE_PATH)
   for content in df['content']:
       text_len.append(len(content))
   print(len([i for i in text_len if i>50]))
   print(len(text_len))


if __name__ =='__main__':
    # print(label_to_id('Efficacy/Side Effects'))
    trans_label('./data/input/tb_question_label_1w.xlsx', TRAIN_SAMPLE_PATH)
    trans_label('./data/input/jd_question_label_2k.xlsx', TEST_SAMPLE_PATH)
    count_text_len()

