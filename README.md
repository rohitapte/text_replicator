# text_replicator
###Teaching RNNs to replicate text

The code is split into 3 parts <br>
1. main.py - sets up the model parameters. there are some arguments passed to the model which can be updated using command line flags.<br>
gpu: which gpu to use (if youhave multiple)
mode: train/demo
and so on.... there are descriptions for each one.
2. data_batcher.py - loads the data and sets up batches
3. LSTMCharacterModel.py - sets up the computational graph for the model

Usage: python main.py. arguments are passed as <br>
--gpu=0 --mode=demo etc
