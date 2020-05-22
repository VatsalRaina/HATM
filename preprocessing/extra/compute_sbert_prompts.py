#! /usr/bin/env python

import argparse
import os
import sys

import numpy as np

from sentence_transformers import SentenceTransformer    # Requires pytorch and python 3.6, so use virtual environment

from sklearn.metrics.pairwise import cosine_similarity

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

    
    train_prompt_path = '/home/alta/relevance/vr311/data_vatsal/BULATS/tfrecords_train/sorted_topics.txt'
    f = open(train_prompt_path, "r")
    samples = f.readlines()
    train_prompts = [line.rstrip('\n') for line in samples]
    f.close()
    print(len(train_prompts))

    model = SentenceTransformer('bert-base-nli-mean-tokens')
    train_prompt_embeddings = model.encode(train_prompts)

    
    save_path = '/home/alta/relevance/vr311/models_min_data/baseline/HATM_bert/ATM_baseline/BERT_prompts'
    path = os.path.join(save_path, 'prompt_embeddings.txt')
    np.savetxt(path, train_prompt_embeddings)
    

if __name__ == '__main__':
    main()
