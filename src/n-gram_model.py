import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm

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

def markov_chain(preDict,ngramDict):
    markovDict={}
    markovProbs={}
    for key in preDict:
        subDict={}
        subList=[]
        markovDict[key]=subDict
        markovProbs[key]=subList
    for key,value in tqdm(ngramDict.items()):
        subDict=markovDict[key[1]]
        subList=markovProbs[key[1]]
        subDict[len(subList)]=key[0]
        subList.append(value/preDict[key[1]])
        markovDict[key[1]]=subDict
        markovProbs[key[1]]=subList
    return markovDict,markovProbs

def generateProbMatrix(text,ngram_size):
    print("generating "+str(ngram_size)+"-grams...")
    ngramDict,preDict=generate_ngrams(text,ngram_size)
    print("generating markov probabilities")
    markovDict, markovProbs=markov_chain(preDict, ngramDict)
    return preDict,markovDict, markovProbs

NGRAM_SIZE=5
print("reading data files...")
text=read_data_files('../paulgraham')
preDict,markovDict,markovProbs=generateProbMatrix(text,NGRAM_SIZE)

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
