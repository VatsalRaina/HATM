#! /usr/bin/env python

import argparse
import os
import sys
import numpy as np
from core.utilities.utilities import text_to_array

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
#commandLineParser.add_argument('prompt_path', type=str,
 #                              help='which should be loaded')
#commandLineParser.add_argument('save_path', type=st
commandLineParser.add_argument('--strip_start_end', action='store_true', help='whether to strip the <s> </s> marks at the beginning and end of prompts in sorted_topics.txt file (used for legacy sorted_topics.txt formatting')


def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/get_bert_dists.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

    
    source_dir = '/home/alta/relevance/vr311/data_vatsal/BULATS'
    prompts_path = os.path.join(source_dir, 'prompts.txt')
    sorted_prompts_path = os.path.join(source_dir, 'tfrecords_train/sorted_topics.txt')

    f = open(prompts_path, "r")
    prompts = f.readlines()
    prompts = [line.rstrip('\n') for line in prompts]
    f.close()

    f = open(sorted_prompts_path, "r")
    sorted_prompts = f.readlines()
    sorted_prompts = [line.rstrip('\n') for line in sorted_prompts]
    f.close()

    prompt_ids = []
    for pr in prompts:
        prompt_ids.append(sorted_prompts.index(pr))

    prompt_resp_ids = [None]*379
    prompt_resp_id_lens = [0]*379
    for i, _ in enumerate(prompt_resp_ids):
        prompt_resp_ids[i] = []
    for i, val in enumerate(prompt_ids):
        prompt_resp_ids[val].append(i)
        prompt_resp_id_lens[val] += 1
 
    prompt_resp_ids = np.asarray(prompt_resp_ids)
    prompt_resp_id_lens = np.asarray(prompt_resp_id_lens, dtype=np.int32)
    
    # Create mask
    data = prompt_resp_ids
    prompt_resp_ids = np.zeros((len(prompt_resp_id_lens), np.max(prompt_resp_id_lens)), dtype=np.int32)
    for i, length in zip(xrange(len(data)), prompt_resp_id_lens):
        prompt_resp_ids[i][0:prompt_resp_id_lens[i]] = data[i]


    save_path = '/home/alta/relevance/vr311/models_min_data/baseline/ATM'
    path = os.path.join(save_path, 'prompt_resp_ids.txt')
    np.savetxt(path, prompt_resp_ids)
    path = os.path.join(save_path, 'prompt_resp_id_lens.txt')
    np.savetxt(path, prompt_resp_id_lens)
    

if __name__ == '__main__':
    main()
