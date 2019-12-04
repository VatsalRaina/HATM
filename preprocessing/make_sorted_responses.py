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

    
    resp_path = '/home/alta/relevance/vr311/data_vatsal/BULATS/responses.txt'
    wlist_path = '/home/alta/relevance/vr311/data_vatsal/input.wlist.index'
    resps, resp_lens = text_to_array(resp_path, wlist_path)
    #print(resps.shape)
      
    save_path = '/home/alta/relevance/vr311/models_min_data/baseline/ATM'
    path = os.path.join(save_path, 'sorted_resps.txt')
    np.savetxt(path, resps)
    path = os.path.join(save_path, 'sorted_resp_lens.txt')
    np.savetxt(path, resp_lens)
    

if __name__ == '__main__':
    main()
