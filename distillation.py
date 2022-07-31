import csv

import numpy as np
import torch
from torch import device
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizer, AdamW
from model.model import StudentBertClassificationModel
import torch.nn as nn
from mydataset import MyDataSet
import transformers

vocab_path = "bert-base-chinese/vocab.txt"
batch_size = 8
train_data_path = "dataset/data_train.csv"
dev_data_path = "dataset/data_dev.csv"
test_data_path = "dataset/data_test.csv"
max_len = 200
teacher_path = 'savedmodel/kd_model_teacher.pkl'
save_path = 'savedmodel/kd_beststudent.pkl'

def load_data(data_path, vocab_path=vocab_path):
    csvFileObj = open(data_path, encoding="utf-8")
    lines = csv.reader(csvFileObj)
    text_list = []
    labels = []

    for line in lines:
        if line[1] != "1" and line[1] != "0" and line[1] != "-1":
            continue;
        label = int(line[1]) + 1

        text = line[0]
        text_list.append(text)
        labels.append(label)
    input_ids, token_type_ids, attention_mask = encode(max_len=150, vocab_path=vocab_path, text_list=text_list)
    labels = torch.tensor(labels)
    data = TensorDataset(input_ids, token_type_ids, attention_mask, labels)
    return data

def encode(text_list, max_len=max_len, vocab_path=vocab_path):
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    tokenizer = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    input_ids = tokenizer['input_ids']
    token_type_ids = tokenizer['token_type_ids']
    attention_mask = tokenizer['attention_mask']
    return input_ids, token_type_ids, attention_mask

#调用load_data加载数据
train_data = load_data(train_data_path)
dev_data = load_data(dev_data_path)
test_data = load_data(test_data_path)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dataset=dev_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


def get_weight(train_loader):
    weight = np.array([0, 0, 0])
    for step, (input_ids, token_type_ids, attention_mask, labels) in enumerate(train_loader):
        weight[labels] += 1
    return weight.sum() / weight.size / weight

def train(model, train_loader, dev_loader, teacher_model=None, temperature=1, save_path=save_path):
    epochs = 2
    model.to(device)
    model.train()

    # 权重可改为数量的倒数
    weight = get_weight(train_loader)
    criterion1 = nn.CrossEntropyLoss(weight=torch.from_numpy(weight).float())
    criterion2 = nn.MSELoss()
    criterion3 = nn.KLDivLoss(reduction="batchmean")

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # 设置模型参数的权重衰减
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer_params = {'lr': 1e-5, 'eps': 1e-6, 'correct_bias': False}
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_params)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-7, patience=5, verbose=True,
                                  threshold=0.0001, eps=1e-08)

    criterion1.to(device)
    criterion2.to(device)
    criterion3.to(device)
    bestAcc = 0
    correct = 0
    total = 0
    print('Training and verification begin')

    model.train()
    for epoch in range(epochs):
        for step, (input_ids, token_type_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids, token_type_ids, attention_mask, labels = input_ids.to(device), token_type_ids.to(
                device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, hidden_states, attentions = model(input_ids, token_type_ids, attention_mask, output_all=True)

            loss = criterion1(logits, labels)
            logits_t, hidden_states_t, attentions_t = teacher_model(input_ids, token_type_ids, attention_mask, output_all=True)

            # predict loss   KL
            loss += criterion3(F.softmax(logits / temperature, dim=1),
                               F.softmax(logits_t / temperature, dim=1))

            # attention 和 hidden_state  loss    MSE
            for i in range(len(hidden_states)):
                loss += criterion2(hidden_states[i], hidden_states_t[i])
            for i in range(len(attentions)):
                loss += criterion2(attentions[i], attentions_t[i])

            __, predict = torch.max(logits.data, 1)
            correct += (predict == labels).sum().item()
            total += labels.size(0)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 20 == 19:
                train_acc = correct / total
                print("Train Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,loss:{:.6f}".format(epoch + 1, epochs, step + 1,
                                                                                          len(train_loader),
                                                                                          train_acc * 100, loss.item()))
            if step % 500 == 499:
                train_acc = correct / total
                # 调用验证函数dev对模型进行验证，并将有效果提升的模型进行保存
                acc = dev(model, dev_loader)
                if bestAcc < acc:
                    bestAcc = acc
                    torch.save(model, save_path)
                print("DEV Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,bestAcc{:.6f}%,dev_acc{:.6f} %,loss:{:.6f}".format(
                    epoch + 1, epochs, step + 1, len(train_loader), train_acc * 100, bestAcc * 100, acc * 100,
                    loss.item()))
        scheduler.step(bestAcc)


def dev(model, dev_loader):
    model.to(device)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for step, (input_ids, token_type_ids, attention_mask, labels) in tqdm(enumerate(dev_loader),
                                                                              desc='Dev Itreation:'):
            input_ids, token_type_ids, attention_mask, labels = input_ids.to(device), token_type_ids.to(
                device), attention_mask.to(device), labels.to(device)
            out_put = model(input_ids, token_type_ids, attention_mask)
            _, predict = torch.max(out_put.data, 1)
            correct += (predict == labels).sum().item()
            total += labels.size(0)
        res = correct / total
        return res


teacher_model = torch.load(teacher_path)

# 实例化模型并进行训练与验证
device = torch.device('cuda:0')

# 实例化模型
model = StudentBertClassificationModel()
# 调用训练函数进行训练与验证
train(model, train_loader, dev_loader, teacher_model=teacher_model)
