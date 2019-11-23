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
import math

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
parser.add_argument('--neg_responses_data_dir', default=None, type=str, help='If specified, responses for negative '
                                                                             'examples will be sampled from this '
                                                                             'directory')

def load_data(data_dir, args):
    """
    Read in dataset response, confidences, prompts, grades and speakers data from a directory.
    :param data_dir: directory with files args.responses_file, args.prompts_file ...etc.
    :return: responses, confidences, prompts, grades, speakers
    """
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
    return responses, confs, prompts, grades, speakers


def main(args):
    if os.path.isdir(args.destination_dir):
        usr_response = raw_input("Destination directory exists. Type 'y' if you want to overwrite it: ")
        if usr_response != 'y':
            print('destination directory {} exists.\nExiting...'.format(args.destination_dir))
            exit()
    else:
        os.makedirs(args.destination_dir)


    # Load the data
    responses, confs, prompts, grades, speakers = load_data(args.data_dir, args)

    if args.neg_responses_data_dir != None:
        # Load the data for negative responses if specified
        print('Negative responses are being loaded from:\t{}'.format(args.neg_responses_data_dir))
        responses_neg, confs_neg, _, grades_neg, speakers_neg = load_data(args.neg_responses_data_dir, args)

    # Cache the command:
    if not os.path.isdir(os.path.join(args.destination_dir, 'CMDs')):
        os.makedirs(os.path.join(args.destination_dir, 'CMDs'))
    with open(os.path.join(args.destination_dir, 'CMDs/preprocessing.cmd'), 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Copy questions
    np.random.seed(args.seed)
    new_data = []

    if args.neg_responses_data_dir == None:
        # Shuffle the responses from data_dir to obtaing negative examples
        shuf_prompts = prompts[:]

        num_on_topic = 0
        num_total = 0

        for sample in xrange(args.samples):
            shuf_prompts = np.random.permutation(shuf_prompts)
            for response, conf, prompt, shuf_prompt, grade, spkr in zip(responses, confs, prompts, shuf_prompts, grades,
                                                                        speakers):
                num_on_topic += 1
                num_total += 2
                if prompt == shuf_prompt:
                    target = 1
                    num_on_topic += 1
                else:
                    target = 0

                new_data.append([response, conf, prompt, str(float(1.0)), spkr, grade])
                new_data.append([response, conf, shuf_prompt, str(float(target)), spkr, grade])

        print('percent relevant:', float(num_on_topic) / float(num_total))
    else:
        # Use the responses from neg_responses_data_dir as negative examples

        # Make sure there are enough negative responses, if not, duplicate them
        num_examples = len(responses)  # Number of original data examples

        if len(responses_neg) < len(responses):
            num_repeats = math.ceil(float(num_examples) / len(responses_neg))
            responses_neg = np.repeat(responses_neg, num_repeats)
            confs_neg = np.repeat(confs_neg, num_repeats)
            grades_neg = np.repeat(grades_neg, num_repeats)
            speakers_neg = np.repeat(speakers_neg, num_repeats)

        for sample in xrange(args.samples):
            # Shuffle the negative example responses
            shuffled_neg_examples = np.random.permutation(list(zip(responses_neg, confs_neg, grades_neg, speakers_neg)))
            # shuffled_neg_examples is a list of lists [[response, conf, grade, speaker], ...] - need to convert back
            responses_neg, confs_neg, grades_neg, speakers_neg = zip(*shuffled_neg_examples)

            # Add the positive examples:
            targets_pos = [str(float(1.0)) for i in range(num_examples)]
            new_data.extend(zip(responses, confs, prompts, targets_pos, speakers, grades))
            # Add the negative examples:
            targets_neg = [str(float(0.0)) for i in range(num_examples)]
            new_data.extend(zip(responses_neg[:num_examples], confs_neg[:num_examples], prompts, targets_neg,
                                speakers_neg[:num_examples], grades_neg[:num_examples]))


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
