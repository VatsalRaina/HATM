#! /usr/bin/env python

import argparse
import os
import sys

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

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

    model = SentenceTransformer('bert-base-nli-mean-tokens')
    prompt_embeddings = model.encode(prompts)

    sbert_dists = np.empty([379, 379])
    
    for i, pr1 in enumerate(prompt_embeddings):
        print(i)
        for j, pr2 in enumerate(prompt_embeddings):
            a = pr1.reshape(1,768)
            b = pr2.reshape(1,768)
            dist = cosine_similarity(a,b)[0][0]
            sbert_dists[i][j] = dist
    
    print(sbert_dists)
    
    save_path = '/home/alta/relevance/vr311/models_min_data/baseline/ATM'
    path = os.path.join(save_path, 'sbert_dists.txt')
    np.savetxt(path, sbert_dists)
    

if __name__ == '__main__':
    main()
