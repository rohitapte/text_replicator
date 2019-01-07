import numpy as np
import os

class LSTMDataObject(object):
    def __init__(self,filepath):
        self.UNK='_UNK_'
        self.UNK_ID=0
        self.char2id={}
        self.char2id[''+chr(9)]=1
        self.char2id[self.UNK]=self.UNK_ID
        self.char2id[''+chr(10)]=127-30
        for i in range(32,127):
            self.char2id[''+chr(i)]=i-30
        self.ALPHASIZE=len(self.char2id)
        self.id2char={}
        for item in self.char2id:
            self.id2char[self.char2id[item]]=item
        self.read_data_files(filepath)

    def encode_text(self,text):
        return [self.char2id.get(item,self.UNK_ID) for item in list(text)]

    def decode_text(self,list_of_ids):
        return ''.join([self.id2char.get(item,self.UNK) for item in list_of_ids])

    def read_data_files(self,directory):
        self.encoded_text=[]
        for root, dirs, files in os.walk(directory):
            for file in files:
                filename=os.path.join(root,file)
                with open(filename,'r') as f:
                    text=f.read()
                    self.encoded_text.extend(self.encode_text(text))

    def generate_one_epoch(self,batch_size,sequence_length,epoch_number):
        data=np.array(self.encoded_text)
        data_len=data.shape[0]
        num_batches=(data_len-1)//(batch_size*sequence_length)
        data_len_rounded=num_batches*batch_size*sequence_length
        x_data=np.reshape(data[0:data_len_rounded],[batch_size,num_batches*sequence_length])
        y_data=np.reshape(data[1:data_len_rounded+1],[batch_size,num_batches*sequence_length])
        for batch in range(num_batches):
            x=x_data[:,batch*sequence_length:(batch+1)*sequence_length]
            y=y_data[:,batch*sequence_length:(batch+1)*sequence_length]
            x=np.roll(x,-epoch_number,axis=0)
            y=np.roll(y,-epoch_number,axis=0)
            yield x,y

#zz=LSTMDataObject()
#count=0
#for x,y in zz.generate_one_epoch(200,30,1):
        #    for i in range(x.shape[0]):
        #print(zz.decode_text(x[i,:].tolist()))
        #print("--------")
        #print(zz.decode_text(y[i,:].tolist()))
    #if i>4:break
    #print("#############################")
    #count += 1
#if count>1:break
