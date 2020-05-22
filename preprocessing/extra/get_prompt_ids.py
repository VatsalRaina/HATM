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

    f = open('/home/alta/relevance/vr311/data_vatsal/LINSK/Andrey/LINSKuns03evl03_ALL_naive/info/unique_prompts.txt', "r")
    temp = f.readlines()
    pr_unique = [line.rstrip('\n') for line in temp]
    f.close()

    f = open('/home/alta/relevance/vr311/data_vatsal/LINSK/Andrey/LINSKuns03evl03_ALL_naive/prompts.txt', "r")
    temp = f.readlines()
    pr_all = [line.rstrip('\n') for line in temp]
    f.close()
    

    ids = np.empty([170072], np.int32)
    
    for i, pr1 in enumerate(pr_all):
        print(i)
        for j, pr2 in enumerate(pr_unique):
            if pr1==pr2:
                ids[i] = j   
 
    
    save_path = '/home/alta/relevance/vr311/data_vatsal/LINSK/Andrey/LINSKuns03evl03_ALL_naive'
    path = os.path.join(save_path, 'pr_ids.txt')
    np.savetxt(path, ids, fmt='%i')
    

if __name__ == '__main__':
    main()
