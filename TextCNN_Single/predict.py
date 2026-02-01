from config import *
from utils import *
from model import *

if __name__ == '__main__':
    id2label, _ = get_label()

    model = TextCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR + 'model_weights_30.pth', map_location=DEVICE))
    # model = torch.load(MODEL_DIR + '30.pth', map_location=DEVICE)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    texts = [
        '可以祛斑吗？',
        '小孩能不能用？',
        '可以祛痘吗，有没有副作用？',
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
    pred_ = torch.argmax(pred, dim=1)

    print([id2label[l] for l in pred_])