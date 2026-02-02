import torch.nn as nn
import torch.nn.functional as F
import torch
from .config import *
from transformers import BertModel


class Multi(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        for name ,param in self.bert.named_parameters():
            param.requires_grad = False
        self.convs = nn.ModuleList([nn.Conv2d(1, NUM_FILTERS, (i, EMBEDDING_DIM)) for i in FILTER_SIZES])
        self.linear = nn.Linear(NUM_FILTERS * 3, NUM_CLASSES)

    def conv_and_pool(self, conv, input):
        out = conv(input)
        out = F.relu(out)
        return F.max_pool2d(out, (out.shape[2], out.shape[3])).squeeze(-1).squeeze(-1)

    def forward(self, input, mask):
        out = self.bert(input, mask)[0].unsqueeze(1)
        out = torch.cat([self.conv_and_pool(conv, out) for conv in self.convs], dim=1)
        return torch.sigmoid(self.linear(out))

    def loss_fn(self, pred, true):
        true = true.float()
        weight = torch.where(true > 0, CLS_WEIGHT_COEF[1], CLS_WEIGHT_COEF[0])
        return F.binary_cross_entropy(pred, true, weight=weight, reduction='mean')


class Single(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        for name ,param in self.bert.named_parameters():
            param.requires_grad = False
        self.convs = nn.ModuleList([nn.Conv2d(1, NUM_FILTERS, (i, EMBEDDING_DIM)) for i in FILTER_SIZES])
        self.linear = nn.Linear(NUM_FILTERS * 3, NUM_CLASSES)

    def conv_and_pool(self, conv, input):
        out = conv(input)
        out = F.relu(out)
        return F.max_pool2d(out, (out.shape[2], out.shape[3])).squeeze(-1).squeeze(-1)

    def forward(self, input, mask):
        out = self.bert(input, mask)[0].unsqueeze(1)
        out = torch.cat([self.conv_and_pool(conv, out) for conv in self.convs], dim=1)
        return self.linear(out)