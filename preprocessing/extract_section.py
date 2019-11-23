#!/usr/bin/python

from __future__ import print_function
import sys
import os
import re
import argparse
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter
import matplotlib

matplotlib.rcParams.update({'font.size': 18})
try:
    import cPickle as pickle
except:
    import pickle

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
# commandLineParser.add_argument ('INPUT', type=str, choices= ['question_name', 'question_index', 'question_text'], default = 'question_name',
#                                help = 'Select input type')
# commandLineParser.add_argument ('OUTPUT', type=str, choices= ['question_names', 'question_index', 'question_text'], default = 'question_index',
#                                help = 'Select output type')
commandLineParser.add_argument('sorted_topics', type=str,
                               help='Input')
commandLineParser.add_argument('key_topics', type=str,
                               help='Input')
commandLineParser.add_argument('alignment_matrix', type=str,
                               help='Input')
commandLineParser.add_argument('unigrams', type=str,
                               help='Input')
commandLineParser.add_argument('auc', type=str, default=None,
                               help='Input')
commandLineParser.add_argument('--fold', type=str, default=None,
                               help='Input')


def main(argv=None):
    args = commandLineParser.parse_args()
    sec_dict = {}
    for section in ['SC', 'SD', 'SE']:
        with open('/home/malinin/dnn_lib/attention_grader/script-maps/scripts-' + section + '-map.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split()
                line = ' '.join(line[2:])
                if not sec_dict.has_key(line):
                    if section == 'SC':
                        sec_dict[line] = 0
                    elif section == 'SD':
                        sec_dict[line] = 1
                    else:
                        sec_dict[line] = 2

    topic_sec_ids = []
    ids_tuples = []
    topics = []
    with open(args.sorted_topics, 'r') as f:
        for line in f.readlines():
            topics.append(line)
            line = line.replace('\n', '')
            line = re.sub(r'^<s>.*</s> <s>', r'<s>', line)
            topic_sec_ids.append(sec_dict[line])
    for topic_sec_id, i in zip(topic_sec_ids, xrange(len(topic_sec_ids))):
        ids_tuples.append((topic_sec_id, i))

    with open('topic_ids.txt', 'w') as f:
        for ids in ids_tuples:
            f.write(str(ids[0]) + '\n')

    key_sec_ids = []
    key_tuples = []
    with open(args.key_topics, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            line = re.sub(r'^<s>.*</s> <s>', r'<s>', line)
            key_sec_ids.append(sec_dict[line])
    for key_sec_id, i in zip(key_sec_ids, xrange(len(key_sec_ids))):
        key_tuples.append((key_sec_id, i))

    sorted_ids_tuples = sorted(ids_tuples, key=itemgetter(0, 1))
    sorted_key_tuples = sorted(key_tuples, key=itemgetter(0, 1))
    alignment_matrix = np.loadtxt(args.alignment_matrix)
    plt.matshow(alignment_matrix, cmap=plt.cm.plasma)
    #  plt.show()
    plt.close()
    sorted_alignment_matrix = np.zeros_like(alignment_matrix)
    for i, x in zip(sorted_ids_tuples, xrange(len(sorted_ids_tuples))):
        for j, y in zip(sorted_key_tuples, xrange(len(sorted_key_tuples))):
            sorted_alignment_matrix[y][x] = alignment_matrix[j[1]][i[1]]
    plt.matshow(sorted_alignment_matrix, cmap=plt.cm.plasma)
    #  plt.show()
    plt.close()
    entropy = np.sum(-alignment_matrix * np.log(alignment_matrix + 1e-8), axis=1)
    _, keys = zip(*sorted(zip(entropy, np.asarray(key_tuples)[:, 1]), key=itemgetter(0)))
    unigrams = np.loadtxt(args.unigrams)
    print
    keys
    with open('new_sorted_topics.txt', 'w') as f:
        for i in keys:
            f.write(topics[i])
    sorted_entropy, _ = zip(*sorted(zip(entropy, np.asarray(key_tuples)[:, 0]), key=itemgetter(1, 0)))
    aucs = np.loadtxt(args.auc)

    plt.plot(sorted_entropy, c='b', lw=3.0)
    plt.axvline(x=52, lw=2.0, ls='--', c='k')
    plt.axvline(x=106, lw=2.0, ls='--', c='k')
    plt.text(19, 5, r'SC', fontsize=22)
    plt.text(70, 5, r'SD', fontsize=22)
    plt.text(200, 5, r'SE', fontsize=22)
    plt.xlim(0, 379)
    plt.ylabel('Entropy')
    plt.xlabel('Topics')
    plt.savefig('attention_entropy.png', bbox_inches='tight')
    # plt.show()
    plt.close()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_ylabel('Entropy')
    ax1.set_xlabel('Fraction of Topics')
    #  plt.ylabel('Entropy')
    # plt.xlim(0,379)
    ax1.plot([i / 379.0 for i in xrange(379)], sorted(entropy, reverse=True), c='r', lw=3.0)
    ax2.set_ylabel('AUC')
    #  ax2.set_xlim(0,76)
    #  ax1.set_xlim(0,379)
    ax2.plot([i / 76.0 for i in xrange(76)], aucs, c='g', ls='--', lw=3.0)
    ax1.legend(['Entropy'], loc=4)
    ax2.legend(['AUC'], loc=3)
    plt.savefig('attention_entropy_auc.png', bbox_inches='tight')
    # plt.show()
    plt.close()

    if args.fold is None:
        plt.savefig('alignment_matrix.png', bbox_inches='tight')
    else:
        plt.savefig('alignment_matrix_fold' + args.fold + '.png', bbox_inches='tight')
    plt.close()
    sys.exit()
    #  print alignment_matrix
    confusion = np.zeros((3, 3), dtype=np.float32)
    for key, i in zip(key_sec_ids, xrange(len(key_sec_ids))):
        for topic, j in zip(topic_sec_ids, xrange(len(topic_sec_ids))):
            confusion[key][topic] += alignment_matrix[i][j]
    print(confusion)  # , np.sum(confusion,axis=1)
    confusion = confusion / np.sum(confusion, axis=1)[:, np.newaxis]
    print(confusion)
    plt.matshow(confusion, cmap=plt.cm.Blues)
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()
    # plt.show()

    return 0


if __name__ == '__main__':
    main()
