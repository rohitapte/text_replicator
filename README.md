# text_replicator
###Teaching RNNs to replicate text
This code is an adaptation of Andrej Karpathy's <a href='http://karpathy.github.io/2015/05/21/rnn-effectiveness/'>The unreasonable effectiveness of Recurrent Neural Networks<\a>. It uses tensorflow with multi layer GRUs and gradient clipping.
The code is split into 3 parts <br>
1. main.py - sets up the model parameters. there are some arguments passed to the model which can be updated using command line flags.<br>
gpu: which gpu to use (if youhave multiple)
mode: train/demo
and so on.... there are descriptions for each one.
2. data_batcher.py - loads the data and sets up batches
3. LSTMCharacterModel.py - sets up the computational graph for the model

Usage: python main.py. arguments are passed as <br>
--gpu=0 --mode=demo etc
