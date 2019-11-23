#! /usr/bin/env python

"""
Expand the data-set by creating negative samples through shuffling. Done to create an evaluation set by shuffling
prompts and responses to generate files with a mix of positive (on-topic) and negative (off-topic) examples.

-----

Generates files:
responses.txt prompts.txt speakers.txt conf.txt sections.txt prompt_ids.txt targets.txt

in the same format as magic_preprocess_raw.py, just with an additional file targets.txt which specifies whether the
prompt response pair is matching (on-topic).
"""
from __future__ import print_function, division
import numpy as np
from math import floor
import sys, os, re
import argparse

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

#from core.utilities.utilities import get_bert_dist
from core.utilities.bertEmbeddings import BertEmbeddings

parser = argparse.ArgumentParser(description='Expand the data-set by creating negative samples through shuffling.')
parser.add_argument('data_dir', type=str,
                    help='absolute path to the directory with the processed responses, prompts, speakers, grades etc. .txt data')
parser.add_argument('destination_dir', type=str,
                    help='absolute path to directory where to save the generated examples.')
parser.add_argument('--samples', type=int, default=10,
                    help='Number of negative samples to create with each response')
parser.add_argument('--responses_file', type=str, default='responses.txt')
parser.add_argument('--prompts_file', type=str, default='prompts.txt')
parser.add_argument('--grades_file', type=str, default='grades.txt')
parser.add_argument('--speakers_file', type=str, default='speakers.txt')
parser.add_argument('--confidences_file', type=str, default='confidences.txt')
parser.add_argument('--seed', type=int, default=1000)


def main(args):
    if os.path.isdir(args.destination_dir):
        usr_response = raw_input("Destination directory exists. Type 'y' if you want to overwrite it: ")
        if usr_response != 'y':
            print('destination directory {} exists.\nExiting...'.format(args.destination_dir))
            exit()
    else:
        os.makedirs(args.destination_dir)

    # Get the paths to the relevant files
    responses_path = os.path.join(args.data_dir, args.responses_file)
    prompts_path = os.path.join(args.data_dir, args.prompts_file)
    grades_path = os.path.join(args.data_dir, args.grades_file)
    speakers_path = os.path.join(args.data_dir, args.speakers_file)
    confidences_path = os.path.join(args.data_dir, args.confidences_file)
    required_files = [responses_path, prompts_path, grades_path, speakers_path, confidences_path]

    # Assert the required files exist
    for path in required_files:
        if not os.path.isfile(path):
            print('File: {} doesn`t exist. Exiting...'.format(path))
            exit()

    # Cache the command:
    if not os.path.isdir(os.path.join(args.destination_dir, 'CMDs')):
        os.makedirs(os.path.join(args.destination_dir, 'CMDs'))
    with open(os.path.join(args.destination_dir, 'CMDs/preprocessing.cmd'), 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Open All the files
    with open(responses_path, 'r') as d:
        responses = [line.replace('\n', '') for line in d.readlines()]
    with open(confidences_path, 'r') as d:
        confs = [line.replace('\n', '') for line in d.readlines()]
    with open(prompts_path, 'r') as q:
        prompts = [line.replace('\n', '') for line in q.readlines()]
    with open(grades_path, 'r') as t:
        grades = [line.replace('\n', '') for line in t.readlines()]
    with open(speakers_path, 'r') as s:
        speakers = [line.replace('\n', '') for line in s.readlines()]

    # Copy questions
    np.random.seed(args.seed)
    shuf_prompts = prompts[:]

    num_on_topic = 0
    num_total = 0

    # Create instance of Bert sentence embedding (BSE)
    bse = BertEmbeddings()
    bse.create_model()
    all_bert_dists = []

    new_data = []
    for sample in xrange(args.samples):
        shuf_prompts = np.random.permutation(shuf_prompts)
        for response, conf, prompt, shuf_prompt, grade, spkr in zip(responses, confs, prompts, shuf_prompts, grades,
                                                                    speakers):
            # Get BERT sentence embeddings and compute distance
            bert_dist = bse.get_bert_dist(prompt, shuf_prompt)
           # print(str(bert_dist))
            if bert_dist > 0.01:
                all_bert_dists.append(bert_dist)

            num_on_topic += 1
            num_total += 2
            if prompt == shuf_prompt:
                target = 1
                num_on_topic += 1
            else:
                target = 0

            new_data.append([response, conf, prompt, str(float(1.0)), spkr, grade])
            if bert_dist > 0.95:
                new_data.append([response, conf, shuf_prompt, str(float(target)), spkr, grade])
            else:
                print('Similar:')
                #print(prompt)
               # print(shuf_prompt)

    """
    mean_dist = sum(all_bert_dists) / len(all_bert_dists)
    print("MEAN")
    print(mean_dist)
    variance_dist = np.var(np.asarray(all_bert_dists))
    print("VARIANCE")
    print(variance_dist)
    # Plot histogram of bert dists
    dist_graph = plt.figure()
    plt.hist(all_bert_dists, bins=20)
    plt.ylabel('Scaled number of prompt pairs')
    plt.xlabel('BERT Euclidean distance between prompt pairs')
    plt.show()
    dist_graph.savefig("hist_angle.png", bbox_inches='tight')
    """

    print('percent relevant:', float(num_on_topic) / float(num_total))
    new_data = list(np.random.permutation(new_data))

    new_responses, new_confs, new_prompts, new_targets, new_speakers, new_grades = zip(*new_data)
    new_responses, new_confs, new_prompts, new_targets, new_speakers, new_grades = map(lambda x: '\n'.join(x),
                                                                                             [new_responses,
                                                                                              new_confs, new_prompts,
                                                                                              new_targets,
                                                                                              new_speakers,
                                                                                              new_grades])

    with open(os.path.join(args.destination_dir, 'grades.txt'), 'w') as g, open(
            os.path.join(args.destination_dir, 'confidences.txt'), 'w') as c, open(
        os.path.join(args.destination_dir, 'responses.txt'), 'w') as r, open(
        os.path.join(args.destination_dir, 'prompts.txt'), 'w') as p, open(
        os.path.join(args.destination_dir, 'targets.txt'), 'w') as t, open(
        os.path.join(args.destination_dir, 'speakers.txt'), 'w') as s:

        r.write(new_responses)
        p.write(new_prompts)
        t.write(new_targets)
        s.write(new_speakers)
        c.write(new_confs)
        g.write(new_grades)

    print('Finished')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
