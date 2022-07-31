import jieba.analyse
s =""
fencilist=[]
stopwordlist=[]
resultlist = []
with open(r"cn_stopwords.txt", 'r', encoding="UTF-8") as f:
    for i in f:
        stopwordlist.append(i)
with open(r"../dataset/data_train.csv",'r',encoding="UTF-8") as test:
    for line in test:
        line.strip()
        fencilist=jieba.cut(line)
        for i in fencilist:
            if(i not in stopwordlist):
                resultlist.append(i)
        with open(r"../dataset/result.csv",'w',encoding="UTF-8") as xx:
            for x in resultlist:
                xx.write(x)
            xx.write('\n')
