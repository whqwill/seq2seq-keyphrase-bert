# seq2seq-keyphrase-bert


The original code is from https://github.com/memray/seq2seq-keyphrase-pytorch


I just add BERT (https://github.com/huggingface/pytorch-pretrained-BERT) to encoder part. I add a new model "Seq2SeqBERT", which uses BERT for encoder and uses GRU for decoder. 


Specifically, I change some code in preprocess.py so that it preprocess data using the tokenizer from BERT, and I add new model in pykp/model.py, relatively I change the beam_search methods in beam_search.py, and there are also some changes in pykp/io.py, train.py, evaluate.py.


But right now, the result is not good, I am still researching it. Here is the experiment report https://github.com/huggingface/pytorch-pretrained-BERT/files/2623599/RNN.vs.BERT.in.Keyphrase.generation.pdf 


Welcome to give me some advice where I did wrong.