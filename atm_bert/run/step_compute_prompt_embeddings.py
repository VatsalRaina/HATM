#! /usr/bin/env python

import argparse
import os
import sys


import context
from atm.atm import AttentionTopicModel
from core.utilities.utilities import text_to_array, IdToWordConverter

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('--load_path', type=str, default='./',
                               help='Specify path to model which should be loaded')
commandLineParser.add_argument('--debug', type=int, default=0,
                               help='Specify path to model which should be loaded')
commandLineParser.add_argument('--epoch', type=str, default=None,
                               help='which should be loaded')
commandLineParser.add_argument('prompt_path', type=str,
                               help='which should be loaded')
commandLineParser.add_argument('wlist_path', type=str,
                               help='which should be loaded')
commandLineParser.add_argument('save_path', type=str,
                               help='which should be loaded')
commandLineParser.add_argument('--strip_start_end', action='store_true', help='whether to strip the <s> </s> marks at the beginning and end of prompts in sorted_topics.txt file (used for legacy sorted_topics.txt formatting')


def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_compute_prompt_embeddings.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

    if args.strip_start_end:
        print("Stripping the first and last word (should correspond to <s> and </s> marks) from the input prompts. Should only be used with legacy dataset formatting")


    prompts, prompt_lens = text_to_array(args.prompt_path, args.wlist_path, strip_start_end=args.strip_start_end)
    # Initialize and Run the Model
    atm = AttentionTopicModel(network_architecture=None,
                              load_path=args.load_path,
                              debug_mode=args.debug,
                              epoch=args.epoch)

    atm.get_prompt_embeddings(prompts, prompt_lens, args.save_path)

if __name__ == '__main__':
    main()