from config import *
from utils import *
from model import *

if __name__ == '__main__':
    id2label, _ = get_label()

    model = TextCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR + 'model_weights_14.pth', map_location=DEVICE))
    # model = torch.load(MODEL_DIR + '20.pth', map_location=DEVICE)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    texts = [
        '这款面霜小孩子可以用吗？会不会有副作用？',
        '我怎么闻着是酒精的味道？正常吗？',
        '请问，春天可以用嘛?',
    ]

    batch_input_ids = []
    batch_mask = []
    for text in texts:
        tokened = tokenizer(text)
        input_ids = tokened['input_ids']
        mask = tokened['attention_mask']

        if len(input_ids) < TEXT_LEN:
            pad_len = (TEXT_LEN - len(input_ids))
            input_ids += [BERT_PAD_ID] * pad_len
            mask += [0] * pad_len
        batch_input_ids.append(input_ids[:TEXT_LEN])
        batch_mask.append(mask[:TEXT_LEN])

    batch_input_ids = torch.tensor(batch_input_ids)
    batch_mask = torch.tensor(batch_mask)

    batch_input_ids = batch_input_ids.to(DEVICE)
    batch_mask = batch_mask.to(DEVICE)

    pred = model(batch_input_ids, batch_mask)

    label_ids = []
    label_names = []

    for row in pred:
        item = np.where(row.cpu().data.numpy() > CLS_PROB_BAR, 1, 0)
        lids = seq2ids(item)
        print(lids)
        label_ids.append(lids)
        label_names.append([id2label[l] for l in lids])
    print(label_ids, label_names)