#! /usr/bin/env python

import argparse
import os
import sys

import tensorflow as tf
import numpy as np

from core.utilities.utilities import text_to_array, get_train_size_from_meta
from hatm.hatm import HierarchicialAttentionTopicModel

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('--valid_size', type=int, default=14188,  # 1034,##28375,
                               help='Specify the validation set size')
commandLineParser.add_argument('--batch_size', type=int, default=100,
                               help='Specify the training batch size')
commandLineParser.add_argument('--learning_rate', type=float, default=1e-3,
                               help='Specify the intial learning rate')
commandLineParser.add_argument('--lr_decay', type=float, default=0.85,
                               help='Specify the learning rate decay rate')
commandLineParser.add_argument('--dropout', type=float, default=1.0,
                               help='Specify the dropout keep probability')
commandLineParser.add_argument('--n_epochs', type=int, default=1,
                               help='Specify the number of epoch to run training for')
commandLineParser.add_argument('--n_samples', type=int, default=1,
                               help='Specify the number of negative samples to take')
commandLineParser.add_argument('--seed', type=int, default=100,
                               help='Specify the global random seed')
commandLineParser.add_argument('--name', type=str, default='model',
                               help='Specify the name of the model')
commandLineParser.add_argument('--debug', type=int, choices=[0, 1, 2], default=0,
                               help='Specify the debug output level')
commandLineParser.add_argument('--load_path', type=str, default='./',
                               help='Specify path to model which should be loaded')
commandLineParser.add_argument('--init', type=str, default=None,
                               help='Specify path to from which to initialize model')
commandLineParser.add_argument('--distortion', type=float, default=1.0,
                               help='Specify whether to use uniform negative sampling')
commandLineParser.add_argument('--epoch', type=str, default=None,
                               help='which should be loaded')
commandLineParser.add_argument('train_data', type=str,
                               help='which should be loaded')
commandLineParser.add_argument('meta_data_path', type=str,
                               help='Path to the meta data file (which contains the dataset size and number of topics).')
commandLineParser.add_argument('valid_data', type=str,
                               help='which should be loaded')
commandLineParser.add_argument('topic_path', type=str,
                               help='which should be loaded')
commandLineParser.add_argument('topic_count_path', type=str,
                               help='which should be loaded')
commandLineParser.add_argument('wlist_path', type=str,
                               help='which should be loaded')
commandLineParser.add_argument('--strip_start_end', action='store_true', help='whether to strip the <s> </s> marks at the beginning and end of prompts in sorted_topics.txt file (used for legacy sorted_topics.txt formatting')



def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_train_attention_grader.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    train_size = get_train_size_from_meta(args.meta_data_path)

    topics, topic_lens = text_to_array(args.topic_path, args.wlist_path, strip_start_end=False)
    # Augmented data
    aug_topics, aug_topic_lens = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/translate/ar1.txt", args.wlist_path, strip_start_end=False)
    aug_topics2, aug_topic_lens2 = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/translate/de1.txt", args.wlist_path, strip_start_end=False)
    aug_topics3, aug_topic_lens3 = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/translate/fr1.txt", args.wlist_path, strip_start_end=False)
    aug_topics4, aug_topic_lens4 = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/translate/greek1.txt", args.wlist_path, strip_start_end=False)
    aug_topics5, aug_topic_lens5 = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/translate/hebrew1.txt", args.wlist_path, strip_start_end=False)
    aug_topics6, aug_topic_lens6 = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/translate/hi1.txt", args.wlist_path, strip_start_end=False)
    aug_topics7, aug_topic_lens7 = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/translate/ja1.txt", args.wlist_path, strip_start_end=False)
    aug_topics8, aug_topic_lens8 = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/translate/ko1.txt", args.wlist_path, strip_start_end=False)
    aug_topics9, aug_topic_lens9 = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/translate/ru1.txt", args.wlist_path, strip_start_end=False)
    aug_topics10, aug_topic_lens10 = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/translate/af1.txt", args.wlist_path, strip_start_end=False)
    aug_topics11, aug_topic_lens11 = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/eda/data/one_aug.txt", args.wlist_path, strip_start_end=False)
    aug_topics12, aug_topic_lens12 = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/eda/data/one_aug2.txt", args.wlist_path, strip_start_end=False)
    aug_topics13, aug_topic_lens13 = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/eda/data/one_aug3.txt", args.wlist_path, strip_start_end=False)
    aug_topics14, aug_topic_lens14 = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/eda/data/one_aug4.txt", args.wlist_path, strip_start_end=False)
    aug_topics15, aug_topic_lens15 = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/eda/data/one_aug5.txt", args.wlist_path, strip_start_end=False)
    aug_topics16, aug_topic_lens16 = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/eda/data/one_aug6.txt", args.wlist_path, strip_start_end=False)
    aug_topics17, aug_topic_lens17 = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/eda/data/one_aug7.txt", args.wlist_path, strip_start_end=False)
    aug_topics18, aug_topic_lens18 = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/eda/data/one_aug8.txt", args.wlist_path, strip_start_end=False)
    aug_topics19, aug_topic_lens19 = text_to_array("/home/alta/relevance/vr311/data_vatsal/BULATS/eda/data/one_aug9.txt", args.wlist_path, strip_start_end=False)



   # if args.strip_start_end:
    #    print("Stripping the first and last word (should correspond to <s> and </s> marks) from the input prompts. Should only be used with legacy dataset formatting")

    bert_dists = np.loadtxt("/home/alta/relevance/vr311/models_min_data/baseline/ATM/bert_dists.txt", dtype=np.float32)
    sbert_weights = np.loadtxt("/home/alta/relevance/vr311/models_min_data/baseline/ATM/sbert_weights.txt", dtype=np.float32)
    arr_unigrams = np.loadtxt("/home/alta/relevance/vr311/models_min_data/baseline/ATM/arr_unigrams.txt", dtype=np.float32)

    atm = HierarchicialAttentionTopicModel(network_architecture=None,
                             seed=args.seed,
                             name=args.name,
                             save_path='./',
                             load_path=args.load_path,
                             debug_mode=args.debug,
                             epoch=args.epoch)

    atm.fit(train_data=args.train_data,
            valid_data=args.valid_data,
            load_path=args.init,
            topics=topics,
            topic_lens=topic_lens,
            aug_topics=aug_topics,
            aug_topic_lens=aug_topic_lens,
            aug_topics2=aug_topics2,
            aug_topic_lens2=aug_topic_lens2,
            aug_topics3=aug_topics3,
            aug_topic_lens3=aug_topic_lens3,
            aug_topics4=aug_topics4,
            aug_topic_lens4=aug_topic_lens4,
            aug_topics5=aug_topics5,
            aug_topic_lens5=aug_topic_lens5,
            aug_topics6=aug_topics6,
            aug_topic_lens6=aug_topic_lens6,
            aug_topics7=aug_topics7,
            aug_topic_lens7=aug_topic_lens7,
            aug_topics8=aug_topics8,
            aug_topic_lens8=aug_topic_lens8,
            aug_topics9=aug_topics9,
            aug_topic_lens9=aug_topic_lens9,
            aug_topics10=aug_topics10,
            aug_topic_lens10=aug_topic_lens10,
            aug_topics11=aug_topics11,
            aug_topic_lens11=aug_topic_lens11,
            aug_topics12=aug_topics12,
            aug_topic_lens12=aug_topic_lens12,
            aug_topics13=aug_topics13,
            aug_topic_lens13=aug_topic_lens13,
            aug_topics14=aug_topics14,
            aug_topic_lens14=aug_topic_lens14,
            aug_topics15=aug_topics15,
            aug_topic_lens15=aug_topic_lens15,
            aug_topics16=aug_topics16,
            aug_topic_lens16=aug_topic_lens16,
            aug_topics17=aug_topics17,
            aug_topic_lens17=aug_topic_lens17,
            aug_topics18=aug_topics18,
            aug_topic_lens18=aug_topic_lens18,
            aug_topics19=aug_topics19,
            aug_topic_lens19=aug_topic_lens19,
            bert_dists=bert_dists,
            bert_weights=sbert_weights,
            arr_unigrams=arr_unigrams,
            unigram_path=args.topic_count_path,
            train_size=train_size,
            learning_rate=args.learning_rate,
            lr_decay=args.lr_decay,
            dropout=args.dropout,
            distortion=args.distortion,
            batch_size=args.batch_size,
            optimizer=tf.train.AdamOptimizer,
            optimizer_params={},
            n_epochs=args.n_epochs,
            n_samples=args.n_samples,
            epoch=0)

    atm.save()


if __name__ == '__main__':
    main()
