# text_replicator
### Teaching RNNs to replicate text
This code is an adaptation of Andrej Karpathy's [The unreasonable effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).<br>
It uses tensorflow with multi layer GRUs and gradient clipping. The model can be set to run continuously (num_epochs=0), and saves the model to file if it finds a better result. Restarting the model will also automatically pick up from where it left off.<br>

One of the inputs to the model is a directory with the input files in text format. I have provided 2 sources in my git - shakespeare and aesop's fables. This should work equally well with code, LATEX, etc.<br>

The code is split into 3 parts <br>
1. main.py - sets up the model parameters. there are some arguments passed to the model which can be updated using command line flags.<br>
gpu: which gpu to use (if youhave multiple)
mode: train/demo
and so on.... there are descriptions for each one.
2. data_batcher.py - loads the data and sets up batches
3. LSTMCharacterModel.py - sets up the computational graph for the model

Usage: python main.py. arguments are passed as <br>
--gpu=0 --mode=demo etc
