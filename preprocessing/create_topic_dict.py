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
commandLineParser.add_argument('section', type=str, choices=['SA', 'SB', 'SC', 'SD', 'SE'], default='SC',
                               help='Which section to process')


def main(argv=None):
    args = commandLineParser.parse_args()
    tmp = {}
    q_dict = {}
    a_dict = {}
    with open('script-maps/scripts-' + args.section + '-map.txt', 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split()

            if line[1][0:2] == 'SA':
                if line[1] in ['SA0016',
                               'SA0030',
                               'SA0032',
                               'SA0034',
                               'SA0041',
                               'SA0042',
                               'SA0056',
                               'SA0062',
                               'SA0066',
                               'SA0071',
                               'SA0079',
                               'SA0081',
                               'SA0121']:
                    value = ' '.join(line[0:2])
                else:
                    value = line[1]
            else:
                if line[1][0:2] == 'SC':
                    line[1] = "SC0001"
                elif line[1][0:2] == 'SD':
                    line[1] = "SD0001"
                value = ' '.join(line[0:2])
            key = ' '.join(line[2:])
            if tmp.has_key(key):
                tmp[key].append(value)
            else:
                tmp[key] = [value]

        for item, i in zip(tmp.items(), xrange(len(tmp.items()))):
            #       print item[0], item[1], i
            q_dict[i] = item[0]
            for key in item[1]:
                a_dict[key] = i

                # print q_dict
        path = 'qa_' + args.section + '_dicts.pickle'
        with open(path, 'wb') as handle:
            pickle.dump([q_dict, a_dict], handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
