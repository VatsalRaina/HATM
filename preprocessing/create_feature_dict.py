#!/usr/bin/python

import sys
import os
import re
import argparse

try:
    import cPickle as pickle
except:
    import pickle

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('dataset', type=str,
                               help='Name of the dataset to process')
commandLineParser.add_argument('acoustic_model', type=str,
                               help='AM type')


def main(argv=None):
    args = commandLineParser.parse_args()
    path = os.path.join('/home/alta/BLTSpeaking/exp-am969/dnn_grader/', args.dataset, args.acoustic_model,
                        'F3/raw_features.txt')
    print path
    features_dict = {}
    with open(path, 'r') as handle:
        for line in handle.readlines():
            line = line.replace('\n', '').split()
            if line[0] == 'f0.f0.mean':
                pass
            else:
                features_dict[line[0]] = ' '.join(line[1:])
    path = 'features_' + args.dataset + '_dict.pickle'
    with open(path, 'wb') as handle:
        pickle.dump(features_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
