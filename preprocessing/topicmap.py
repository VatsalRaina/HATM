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
commandLineParser.add_argument('--MERGE_SE', type=bool, default=False,
                               help='Merge SE Question Blurb with sub questions')
commandLineParser.add_argument('INPUT', type=str, choices=['question_name', 'question_index', 'question_text'],
                               default='question_name',
                               help='Select input type')
commandLineParser.add_argument('OUTPUT', type=str, choices=['question_names', 'question_index', 'question_text'],
                               default='question_index',
                               help='Select output type')
commandLineParser.add_argument('input', type=str,
                               help='Input')


def main(argv=None):
    args = commandLineParser.parse_args()
    tmp = {}
    inv_tmp = {}
    q_dict = {}
    a_dict = {}
    for section in ['SA', 'SB', 'SC', 'SD', 'SE']:
        with open('./script-maps/scripts-' + section + '-map.txt', 'r') as f:
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
                        #               print 'here'
                        value = ' '.join(line[0:2])
                    else:
                        value = line[1]
                else:
                    if line[1][0:2] == 'SC':
                        line[1] = "SC0001"
                    elif line[1][0:2] == 'SD':
                        line[1] = "SD0001"
                    elif line[1][0:2] == 'SE' and line[1][-2:] not in ['01', '02', '03', '04', '05']:
                        line[1] = 'SE0006'
                    value = ' '.join(line[0:2])
                key = ' '.join(line[2:])
                if tmp.has_key(key):
                    tmp[key].append(value)
                else:
                    tmp[key] = [value]
                inv_tmp[value] = key

    text_index_dict = {}
    text_names_dict = {}
    names_index_dict = {}
    names_text_dict = {}
    index_text_dict = {}
    index_names_dict = {}
    names_names_dict = {}
    for item, i in zip(tmp.items(), xrange(len(tmp.items()))):
        # item 0 is text item 1 is name, i is index.
        text_index_dict[item[0]] = i
        text_names_dict[item[0]] = item[1]
        text = item[0]
        for name in item[1]:
            if args.MERGE_SE:
                if len(name) > 6 and name.split()[1][0:2] == 'SE':
                    try:
                        key = name.split()[0] + ' SE0006'
                        text = ' '.join([inv_tmp[key], item[0]])
                    except:
                        print 'SE0006 missing in', item[1]
            names_names_dict[name] = item[1]
            names_text_dict[name] = text
            names_index_dict[name] = i
        index_text_dict[i] = text
        index_names_dict[i] = item[1]
        # print item[0], item[1], i
        q_dict[i] = text
        for key in item[1]:
            a_dict[key] = i

    if args.INPUT == 'question_name':
        if args.OUTPUT == 'question_names':
            print names_names_dict[args.input]
        elif args.OUTPUT == 'question_text':
            print names_text_dict[args.input]
        elif args.OUTPUT == 'question_index':
            print names_index_dict[args.input]
    elif args.INPUT == 'question_text':
        if args.OUTPUT == 'question_names':
            print text_names_dict[args.input]
        elif args.OUTPUT == 'question_text':
            print args.input
        elif args.OUTPUT == 'question_index':
            print text_index_dict[args.input]
    elif args.INPUT == 'question_index':
        if args.OUTPUT == 'question_names':
            print index_names_dict[int(args.input)]
        elif args.OUTPUT == 'question_text':
            print index_text_dict[int(args.input)]
        elif args.OUTPUT == 'question_index':
            print args.input

    path = 'qa_ALL_dicts.pickle'
    with open(path, 'wb') as handle:
        pickle.dump([q_dict, a_dict], handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
