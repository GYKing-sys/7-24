import csv

import numpy as np
import torch
from torch import device
from torch.autograd.grad_mode import F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizer, AdamW
from model import BertModel
import torch.nn as nn
from mydataset import MyDataSet
import transformers


vocab_path="bert-base-chinese/vocab.txt"
batch_size = 16
train_data_path= "dataset/data_train.csv"
dev_data_path= "dataset/data_dev.csv"
test_data_path= "dataset/data_test.csv"



def encoder(max_len,vocab_path,text_list):
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    tokenizer = tokenizer(
        text_list,
        padding = True,
        truncation = True,
        max_length = max_len,
        return_tensors='pt'  # 返回的类型为pytorch tensor
        )
    input_ids = tokenizer['input_ids']
    token_type_ids = tokenizer['token_type_ids']
    attention_mask = tokenizer['attention_mask']
    return input_ids,token_type_ids,attention_mask


def load_data(data_path, vocab_path=vocab_path):
    csvFileObj = open(data_path,encoding="utf-8")
    lines = csv.reader(csvFileObj)
    text_list = []
    labels = []

    for line in lines:
        if line[1]!= "1" and line[1]!= "0" and line[1]!= "-1":
            continue;
        #将-1,0,1转为0,1,2
        label = int(line[1])+1
        text = line[0]
        text_list.append(text)
        labels.append(label)
    input_ids,token_type_ids,attention_mask = encoder(max_len=150, vocab_path=vocab_path, text_list=text_list)
    labels = torch.tensor(labels)
    #封装为Tensor
    data = TensorDataset(input_ids,token_type_ids,attention_mask,labels)
    return data


#调用load_data函数，将数据加载为Tensor形式
train_data = load_data(train_data_path)
dev_data = load_data(dev_data_path)
test_data = load_data(test_data_path)

#将训练数据和测试数据进行DataLoader实例化
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dataset=dev_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

def get_weight(train_loader):
    weight=np.array([0,0,0])
    for step, (input_ids, token_type_ids, attention_mask, labels) in enumerate(train_loader):
        weight[labels]+=1
    return 1/weight

def train(model,train_loader,dev_loader) :

    epochs=10

    model.to(device)

    model.train()
    criterion = nn.CrossEntropyLoss()
    #权重可改为数量的倒数
    weight=get_weight(train_loader)
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weight).float(),
        size_average=True)
    criterion.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer,mode='max',factor=0.5,min_lr=1e-7, patience=5,verbose= True, threshold=0.0001, eps=1e-08)

    bestAcc = 0
    correct = 0
    total = 0
    print('Training and verification begin!')

    model.train()
    for epoch in range(epochs):
        for step, (input_ids,token_type_ids,attention_mask,labels) in enumerate(train_loader):
            input_ids, token_type_ids, attention_mask, labels = input_ids.to(device), token_type_ids.to(
                device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, token_type_ids, attention_mask)
            loss = criterion(logits, labels)

            __, predict = torch.max(logits.data, 1)
            correct += (predict == labels).sum().item()
            total += labels.size(0)

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            if step % 20 == 19:
                train_acc = correct / total
                print("Train Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,loss:{:.6f}".format(epoch + 1, epochs, step + 1, len(train_loader),train_acc*100,loss.item()))
            if step % 500 == 499:
                train_acc = correct / total
                #调用验证函数dev对模型进行验证，并将有效果提升的模型进行保存
                acc = dev(model, dev_loader)
                if bestAcc < acc:
                    bestAcc = acc
                    save_path = 'savedmodel/model1.pkl'
                    torch.save(model, save_path)
                print("DEV Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,bestAcc{:.6f}%,dev_acc{:.6f} %,loss:{:.6f}".format(epoch + 1, epochs, step + 1, len(train_loader),train_acc*100,bestAcc*100,acc*100,loss.item()))
        scheduler.step(bestAcc)

def dev(model,dev_loader):
    #将模型放到服务器上
    model.to(device)
    #设定模式为验证模式
    model.eval()
    #设定不会有梯度的改变仅作验证
    with torch.no_grad():
        correct = 0
        total = 0
        for step, (input_ids,token_type_ids,attention_mask,labels) in tqdm(enumerate(dev_loader),desc='Dev Itreation:'):
            input_ids,token_type_ids,attention_mask,labels=input_ids.to(device),token_type_ids.to(device),attention_mask.to(device),labels.to(device)
            out_put = model(input_ids,token_type_ids,attention_mask)

            _, predict = torch.max(out_put.data, 1)
            correct += (predict==labels).sum().item()
            total += labels.size(0)
        res = correct / total
        return res

#实例化模型并进行训练与验证
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')


#实例化模型
model = BertModel()
#调用训练函数进行训练与验证
train(model,train_loader,dev_loader)

