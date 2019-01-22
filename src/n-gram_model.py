import os
from collections import defaultdict
import numpy as np

def read_data_files(directory):
    all_text=''
    for root, dirs, files in os.walk(directory):
        for file in files:
            filename = os.path.join(root, file)
            with open(filename, 'r') as f:
                text = f.read()
                all_text+=text.rstrip()+' '
    return all_text

def generate_ngrams(text,n):
    ngramDict=defaultdict(float)
    preDict=defaultdict(float)
    total=0.0
    for i in range(len(text)-n+1):
        sTemp=text[i:i+n]
        sPredChar=sTemp[-1]
        sPreChar=sTemp[:-1]
        ngramDict[(sPredChar,sPreChar)]+=1.0
        preDict[sPreChar]+=1.0
        total+=1
    for item in ngramDict:
        ngramDict[item]/=total
    for item in preDict:
        preDict[item]/=total
    return ngramDict,preDict

def markob_chain(preDict,ngramDict):
    markovDict={}
    markovProbs={}
    for item in preDict:
        subDict={}
        subList=[]
        index=0
        for subitem in ngramDict:
            if subitem[1]==item:
                subList.append(ngramDict[subitem]/preDict[item])
                subDict[index]=subitem[0]
                index+=1
        markovDict[item]=subDict
        markovProbs[item]=subList
    return markovDict,markovProbs

NGRAM_SIZE=10

print("reading data files...")
text=read_data_files('../shakespeare')
print("generating "+str(NGRAM_SIZE)+"-grams...")
ngramDict,preDict=generate_ngrams(text,NGRAM_SIZE)
print("generating markov probabilities")
markovDict,markovProbs=markob_chain(preDict,ngramDict)

preList=[]
indexToPre={}
for i,item in enumerate(preDict):
    preList.append(preDict[item])
    indexToPre[i]=item

p=np.array(preList)
#p[np.argsort(p)[:-TOP_PROBS]] = 0
#p = p / np.sum(p)
id=np.random.choice(len(preList),1,p=p)[0]
currentText=""
currentText+=indexToPre[id]
nextText=indexToPre[id]
print(currentText,end="")
count=0
while True:
    subProbs=markovProbs[nextText]
    subDict=markovDict[nextText]
    p = np.array(subProbs)
    id = np.random.choice(len(subProbs),1,p=p)[0]
    sTemp=subDict[id]
    currentText+=sTemp
    nextText=currentText[-(NGRAM_SIZE-1):]
    print(sTemp,end="")
    count += 1
    if count % 100 == 0:
        print("")
