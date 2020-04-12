#! /usr/bin/env python

import argparse
import os
import sys

import numpy as np
# Use python 3
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
#commandLineParser.add_argument('prompt_path', type=str,
 #                              help='which should be loaded')
#commandLineParser.add_argument('save_path', type=str,
 #                              help='which should be loaded')
commandLineParser.add_argument('--strip_start_end', action='store_true', help='whether to strip the <s> </s> marks at the beginning and end of prompts in sorted_topics.txt file (used for legacy sorted_topics.txt formatting')

def convert(lst):
    return (lst.split())

def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/get_bert_dists.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

    
    prompt_path = '/home/alta/relevance/vr311/data_vatsal/BULATS/Andrey/seen_seen/prompts.txt'
    f = open(prompt_path, "r")
    train_prompts = f.readlines()
    #train_prompts = [line.rstrip('\n') for line in samples]
    f.close()
    print(len(train_prompts))

   # prompts = np.asarray(train_prompts)

    stop_words = stopwords.words('english')
    content_prompts = []
    i=0
    for pr in train_prompts:
        print(i)
        i+=1
        content_prompts.append(' '.join([word for word in convert(pr) if word.casefold() not in stop_words or len(convert(pr)) < 9]))
        #print(' '.join([word for word in convert(pr) if word not in stop_words]))

    save_path = '/home/alta/relevance/vr311/data_vatsal/BULATS/Andrey/seen_seen/content_words/prompts.txt'
    out_prompts = open(save_path, 'w')
    out_prompts.writelines([line + '\n' for line in content_prompts])
    out_prompts.close() 


    response_path = '/home/alta/relevance/vr311/data_vatsal/BULATS/Andrey/seen_seen/responses.txt'
    f = open(response_path, "r")
    train_responses = f.readlines()
    #train_responses = [line.rstrip('\n') for line in samples]
    f.close()
    print(len(train_responses))
    
    #responses = np.asarray(train_responses)
    
    content_responses = []
    i=0
    for rp in train_responses:
        print(i)
        i+=1
        content_responses.append(' '.join([word for word in convert(rp) if word.casefold() not in stop_words or len(convert(rp)) < 13]))

    save_path = '/home/alta/relevance/vr311/data_vatsal/BULATS/Andrey/seen_seen/content_words/responses.txt'
    out_responses = open(save_path, 'w')
    out_responses.writelines([line +'\n' for line in content_responses])
    out_responses.close()

    """
    save_path = '/home/alta/relevance/vr311/models_min_data/baseline/ATM'
    path = os.path.join(save_path, 'sbert_dists.txt')
    np.savetxt(path, sbert_dists)
    """

if __name__ == '__main__':
    main()
