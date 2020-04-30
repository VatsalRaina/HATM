#! /usr/bin/env python

import argparse
import os
import sys

import numpy as np


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

    train_prompt_embeddings = np.loadtxt('/home/alta/relevance/vr311/models_min_data/baseline/HATM/com1/BULATS_prompts/prompt_embeddings.txt', dtype=np.float32)  
    test_prompt_embeddings = np.loadtxt('/home/alta/relevance/vr311/models_min_data/baseline/HATM/com1/LINSK_prompts/prompt_embeddings.txt', dtype=np.float32)

    sbert_dists = np.empty([11, 379])
    
    for i, pr1 in enumerate(test_prompt_embeddings):
        print(i)
        for j, pr2 in enumerate(train_prompt_embeddings):
            a = pr1.reshape(1,400)
            b = pr2.reshape(1,400)
            dist = cosine_similarity(a,b)[0][0]
            sbert_dists[i][j] = dist

    min_dists = np.amin(sbert_dists, axis=1)    
    print(min_dists)
    
    """
    save_path = '/home/alta/relevance/vr311/models_min_data/baseline/ATM'
    path = os.path.join(save_path, 'sbert_dists.txt')
    np.savetxt(path, sbert_dists)
    """

if __name__ == '__main__':
    main()
