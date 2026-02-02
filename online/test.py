import os
import pandas as pd
from models.classification.functions import predict

if __name__ == '__main__':
    file_path = os.path.join(os.path.dirname(__file__), 'test_question_sample.csv')
    df = pd.read_csv(file_path, usecols=['content', 'labels'], dtype={'labels': str})

    # 分段处理
    i = 0
    pred_labels = []

    while True:
        j = i + 100
        texts = df[i:j]['content'].tolist()
        if not texts:
            break

        label_ids, label_texts = predict(texts)
        pred_labels += ['|'.join([str(id) for id in ids]) for ids in label_ids]

        i = j

    cnt = 0
    for x, y in zip(pred_labels, df['labels'].tolist()):
        if x == y:
            cnt += 1
    print('模型准确率：', round(cnt / len(df['labels']), 4))
