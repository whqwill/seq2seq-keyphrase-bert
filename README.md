# seq2seq-keyphrase-bert


The original code is from https://github.com/memray/seq2seq-keyphrase-pytorch, which is used to do keyphrase generation using seq2seq with attention model. 


Recently BERT (https://arxiv.org/abs/1810.04805) is very popular for many NLP taks, so I add BERT (https://github.com/huggingface/pytorch-pretrained-BERT) to the encoder part of the seq2seq model. I add a new model "Seq2SeqBERT", which uses BERT for encoder and uses GRU for decoder. 


Specifically, I change some code in preprocess.py so that it preprocesses data using the tokenizer from BERT, and I add new model in pykp/model.py, relatively I change the beam_search methods in beam_search.py, and there are also some changes in pykp/io.py, train.py, evaluate.py.


But right now, the result is not good, I am still researching it. Here is the experiment report https://github.com/huggingface/pytorch-pretrained-BERT/files/2623599/RNN.vs.BERT.in.Keyphrase.generation.pdf 


Welcome to give me some advice where I did wrong.


You can use train.py to train seq2seq model. To use BERT, just set the encoder_type to 'bert', and it will initialize with "Seq2SeqBERT". The encoder details are also in Seq2SeqBERT model in pykp/model.py.
