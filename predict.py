# -*- coding: utf-8 -*-
import os
import sys
import argparse
from evaluate import evaluate_beam_search
import logging
import numpy as np

import config
import utils

import torch
import torch.nn as nn
from torch import cuda

from beam_search import SequenceGenerator
from pykp.dataloader import KeyphraseDataLoader
from train import load_data_vocab, init_model, init_optimizer_criterion
from utils import Progbar, plot_learning_curve_and_write_csv

import pykp
from pykp.io import KeyphraseDatasetTorchText, KeyphraseDataset

from pytorch_pretrained_bert.tokenization import BertTokenizer

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

logger = logging.getLogger()

def load_vocab_and_testsets(opt):
    logger.info("Loading vocab from disk: %s" % (opt.vocab))
    word2id, id2word, vocab = torch.load(opt.vocab, 'rb')
    opt.word2id = word2id
    opt.id2word = id2word
    opt.vocab = vocab
    if not opt.decode_old:
        opt.vocab_size = len(word2id)
    logger.info('#(vocab)=%d' % len(word2id))
    logger.info('#(vocab used)=%d' % len(word2id))

    pin_memory = torch.cuda.is_available() and opt.useGpu
    test_one2many_loaders = []

    for testset_name in opt.test_dataset_names:
        logger.info("Loading test dataset %s" % testset_name)

        print ("test_dataset_names")
        print (opt.test_dataset_names)
        print ("testset_name")
        print (testset_name)
        print ()

        testset_path = os.path.join(opt.test_dataset_root_path, testset_name, testset_name + '.test.one2many.pt')
        test_one2many = torch.load(testset_path, 'wb')
        test_one2many_dataset = KeyphraseDataset(test_one2many, word2id=word2id, id2word=id2word, type='one2many', include_original=True)
        test_one2many_loader = KeyphraseDataLoader(dataset=test_one2many_dataset,
                                                   collate_fn=test_one2many_dataset.collate_fn_one2many if opt.useCLF else test_one2many_dataset.collate_fn_one2many_noBeginEnd,
                                                   num_workers=opt.batch_workers,
                                                   max_batch_example=opt.beam_search_batch_example,
                                                   max_batch_pair=opt.beam_search_batch_size,
                                                   pin_memory=pin_memory,
                                                   shuffle=False)

        test_one2many_loaders.append(test_one2many_loader)
        logger.info('#(test data size:  #(one2many pair)=%d, #(one2one pair)=%d, #(batch)=%d' % (len(test_one2many_loader.dataset), test_one2many_loader.one2one_number(), len(test_one2many_loader)))
        logger.info('*' * 50)

    return test_one2many_loaders, word2id, id2word, vocab


def main():
    opt = config.init_opt(description='predict.py')

    opt.data = 'data3/kp20k/kp20k'
    opt.vocab = 'data3/kp20k/kp20k.vocab.pt'
    #opt.train_from = 'exp/kp20k.ml.copy.20181129-193506/model/kp20k.ml.copy.epoch=1.batch=20000.total_batch=20000.model'
    opt.train_from = 'exp/kp20k.ml.copy.20181128-153121/model/kp20k.ml.copy.epoch=2.batch=15495.total_batch=38000.model'

    opt.useGpu = 0
    opt.encoder_type = 'rnn'

    opt.useCLF = False

    if opt.encoder_type.startswith('transformer'):
        opt.batch_size = 32
        opt.d_inner = 2048
        opt.enc_n_layers = 4
        opt.dec_n_layers = 2
        opt.n_head = 8
        opt.d_k = 64
        opt.d_v = 64
        opt.d_model = 512
        opt.word_vec_size = 512
        opt.run_valid_every = 5000000
        opt.save_model_every = 20000
        opt.decode_old = True
        # opt.copy_attention = False
    elif opt.encoder_type.startswith('bert'):
        opt.useOnlyTwo = False
        opt.avgHidden = True
        opt.useZeroDecodeHidden = False
        opt.useSameEmbeding = False
        opt.batch_size = 10
        opt.max_sent_length = 10
        opt.run_valid_every = 20000
        opt.decode_old = False
        opt.beam_search_batch_size = 10
        opt.bert_model = 'bert-base-uncased'
        opt.tokenizer = BertTokenizer.from_pretrained(opt.bert_model)
        if opt.encoder_type == 'bert_low':
            opt.copy_attention = False
    else:
        opt.enc_layers = 2
        opt.bidirectional = True
        opt.decode_old = True

    logger = config.init_logging('predict', opt.exp_path + '/output.log', redirect_to_stdout=False)

    logger.info('EXP_PATH : ' + opt.exp_path)

    logger.info('Parameters:')
    [logger.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    logger.info('======================  Checking GPU Availability  =========================')
    if torch.cuda.is_available() and opt.useGpu:
        if isinstance(opt.gpuid, int):
            opt.gpuid = [opt.gpuid]
        logger.info('Running on %s! devices=%s' % ('MULTIPLE GPUs' if len(opt.gpuid) > 1 else '1 GPU', str(opt.gpuid)))
    else:
        logger.info('Running on CPU!')

    try:
        test_data_loaders, word2id, id2word, vocab = load_vocab_and_testsets(opt)
        model = init_model(opt)
        if torch.cuda.is_available() and opt.useGpu:
            model.cuda()

        generator = SequenceGenerator(model,
                                      opt.word_vec_size if opt.encoder_type == 'transformer' else opt.vocab_size,
                                      eos_id=opt.word2id[pykp.io.EOS_WORD],
                                      beam_size=opt.beam_size,
                                      max_sequence_length=opt.max_sent_length,
                                      useGpu=opt.useGpu
                                      )

        for testset_name, test_data_loader in zip(opt.test_dataset_names, test_data_loaders):
            logger.info('Evaluating %s' % testset_name)
            evaluate_beam_search(generator, test_data_loader, opt,
                                 title='test_%s' % testset_name,
                                 predict_save_path=opt.pred_path + '/%s_test_result/' % (testset_name))

    except Exception as e:
        logger.error(e, exc_info=True)

if __name__ == '__main__':
    main()
