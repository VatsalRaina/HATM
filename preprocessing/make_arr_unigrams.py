#! /usr/bin/env python

import argparse
import os
import sys

import numpy as np

from core.utilities.bertEmbeddings import BertEmbeddings
from core.utilities.utilities import text_to_array_bert

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
#commandLineParser.add_argument('prompt_path', type=str,
 #                              help='which should be loaded')
#commandLineParser.add_argument('save_path', type=str,
 #                              help='which should be loaded')
commandLineParser.add_argument('--strip_start_end', action='store_true', help='whether to strip the <s> </s> marks at the beginning and end of prompts in sorted_topics.txt file (used for legacy sorted_topics.txt formatting')


def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/get_bert_dists.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

    
    uni_path = '/home/alta/relevance/vr311/data_vatsal/BULATS/tfrecords_train/unigrams.txt'
    f = open(uni_path, "r")
    samples = f.readlines()
    _unis = [line.rstrip('\n') for line in samples]
    f.close()
    print(len(_unis))

    # Keep after comma only
    unis = []
    for line in _unis:
        unis.append(line.split(',',1)[1])

    sing_uni = np.empty(379)
    for count, elm in enumerate(unis):
        res = ''.join(filter(lambda i: i.isdigit(), elm))
        sing_uni[count] = res

    print(sing_uni)
    # Normalize
    norm = np.linalg.norm(sing_uni, 1)
    sing_uni_norm = sing_uni / norm
    arr_unigrams = np.tile(sing_uni_norm, (379,1))
    print(arr_unigrams.shape)

    
    
    save_path = '/home/alta/relevance/vr311/models_min_data/baseline/ATM'
    path = os.path.join(save_path, 'arr_unigrams.txt')
    np.savetxt(path, arr_unigrams)


if __name__ == '__main__':
    main()
