import pandas as pd
from config import *
from utils import *

def count_text_len():
    question_text_len = []
    answer_text_len = []
    df = pd.read_excel(TRAIN_SAMPLE_PATH)
    print(len(df))
    df.dropna(subset=['questionContent', 'answerContent', 'sentiment'], how='any', inplace=True)
    print(len(df))
    for id, row in df.iterrows():
        question_text_len.append(len(row['questionContent']))
        answer_text_len.append(len(row['answerContent']))
    print(len([i for i in question_text_len if i>50]))
    print(len([i for i in answer_text_len if i>50]))

if __name__ == '__main__':
    count_text_len()