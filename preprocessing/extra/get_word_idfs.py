#! /usr/bin/env python

import argparse
import os
import sys

from core.utilities.utilities import text_to_array

import numpy as np

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

# Get document of all training (unshuffled) responses
# Convert each word to an ID
# Initialise an np array of dimension of the largest possible ID for a word - call it word_freqs
# Initialise each element in array to 0
# Loop through list of all response word IDs and increment corresponding position in word_freqs by 1 for each ID encountered
# Loop through word_freqs: if element is > 0, set to 1/element; if element is = 0, set to 1.5 (this is the value set by lots of real algorithms)
# Save word_freqs array as a numpy array that can be loaded by step_train_simGrid.py and converted to tf tensor and then have tf.gather to applied to it for a list of word IDs.
    
    response_path = '/home/alta/relevance/vr311/data_vatsal/BULATS/content_words/responses.txt'
    wlist_path = '/home/alta/relevance/vr311/data_vatsal/BULATS/content_words/tfrecords_train/input.wlist.index'
    data, _ = text_to_array(response_path, wlist_path, strip_start_end=False)
    
# Note, input.wlist.index starts word IDs at 0 but all word IDs in 'data' start at 1 (i.e. corresponds to the line number in input.wlist.index instead), so a word ID of 0 is impossible

    word_freqs = np.zeros(62415+1)
    print('GOT HERE')

    print(data[:2])
    

    for w_id in np.nditer(data):
        word_freqs[w_id] += 1

    print('WOW, HERE NOW')
    
    for w in np.nditer(word_freqs, op_flags = ['readwrite']):
        if w > 0:
            w[...] = 1/w
        else:
            w[...] = 1.5

    print('PHEW')
    # Word ID of 0 is impossible
    word_freqs[0] = -1 

    np.savetxt('/home/alta/relevance/vr311/data_vatsal/BULATS/content_words/idf.txt', word_freqs) 
    
    sort_word_freqs = np.sort(word_freqs)
    top_ten = sort_word_freqs[:10]
    for val in np.nditer(top_ten):
        print(np.where(val >= word_freqs ))

if __name__ == '__main__':
    main()
