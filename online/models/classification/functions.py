from .config import *
from .utils import *
from .model import *
from transformers import BertTokenizer
import numpy as np


def predict(texts):
    id2label, _ = get_label()

    multi_model = Multi().to(DEVICE)
    multi_model.load_state_dict(torch.load(MODEL_DIR + 'model_weights_multi_99.pth', map_location=DEVICE))

    single_model = Single().to(DEVICE)
    single_model.load_state_dict(torch.load(MODEL_DIR + 'model_weights_single_99.pth', map_location=DEVICE))

    # model = torch.load(MODEL_DIR + '20.pth', map_location=DEVICE)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

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

    pred = multi_model(batch_input_ids, batch_mask)

    label_ids = []
    label_names = []

    for i, row in enumerate(pred):
        item = np.where(row.cpu().data.numpy() > CLS_PROB_BAR, 1, 0)
        lids = seq2ids(item)

        if not lids:
            single_input_ids = torch.tensor([batch_input_ids[i].tolist()])
            single_mask = torch.tensor([batch_mask[i].tolist()])
            single_pred = single_model(single_input_ids, single_mask)
            lids = torch.argmax(single_pred, dim=1).tolist()

        label_ids.append(lids)
        label_names.append([id2label[l] for l in lids])
    return label_ids, label_names