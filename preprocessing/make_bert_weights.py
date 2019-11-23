#! /usr/bin/env python

import argparse
import os
import sys

import tensorflow as tf
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

    
    prompt_path = '/home/alta/relevance/vr311/data_vatsal/BULATS/tfrecords_train/sorted_topics.txt'
    f = open(prompt_path, "r")
    samples = f.readlines()
    prompts = [line.rstrip('\n') for line in samples]
    f.close()
    print(len(prompts))

    # Create instance of Bert sentence embedding (BSE)
    bse = BertEmbeddings()
    bse.create_model()

    bert_weights = np.empty([379, 379])

    for i, pr1 in enumerate(prompts):
        print(i)
        tot = 0
        for j, pr2 in enumerate(prompts):
            dist = abs(bse.get_bert_dist(pr1, pr2))
            tot += 1/dist
            bert_weights[i][j] = 1/dist
        for j, pr2 in enumerate(prompts):
            # Normalise
            bert_weights[i][j] = bert_weights[i][j]/tot
    
    print(bert_weights)
    
    save_path = '/home/alta/relevance/vr311/models_min_data/baseline/ATM'
    path = os.path.join(save_path, 'bert_weights.txt')
    np.savetxt(path, bert_weights)


if __name__ == '__main__':
    main()
