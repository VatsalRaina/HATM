#! /usr/bin/env python

"""
Create a k-fold split of the data for teacher-student ensemble training where the student learns on data unseen to the ensemble.
"""
from __future__ import print_function, division
import numpy as np
from math import floor
import sys, os, re
import argparse

parser = argparse.ArgumentParser(description='Create k new k-fold split datasets that can be used for generating '
                                             'teacher data on unseen prompts from a single dataset.')
parser.add_argument('data_dir', type=str,
                    help='absolute path to the directory with the processed responses, prompts, speakers, grades etc. .txt data')
parser.add_argument('destination_dir', type=str,
                    help='absolute path to directory where to save the generated examples.')
parser.add_argument('k', type=int, help='how many splits to make')
parser.add_argument('--responses_file', type=str, default='responses.txt')
parser.add_argument('--prompts_file', type=str, default='prompts.txt')
parser.add_argument('--grades_file', type=str, default='grades.txt')
parser.add_argument('--speakers_file', type=str, default='speakers.txt')
parser.add_argument('--confidences_file', type=str, default='confidences.txt')


class Example(object):
    def __init__(self, response, prompt, grade, speaker, confidence):
        self.response = response
        self.prompt = prompt
        self.grade = grade
        self.speaker = speaker
        self.confidence = confidence


class ExamplesSet(object):
    def __init__(self, responses, prompts, grades, speakers, confidences):
        assert len(set([len(responses), len(confidences), len(prompts), len(grades), len(speakers)])) == 1

        self.responses = responses
        self.prompts = prompts
        self.grades = grades
        self.speakers = speakers
        self.confidences = confidences

        self.size = len(grades)

    def __getitem__(self, key):
        if type(key) is int:
            return Example(self.responses[key], self.prompts[key], self.grades[key], self.speakers[key],
                           self.confidences[key])
        elif type(key) is slice:
            return ExamplesSet(self.responses[key], self.prompts[key], self.grades[key], self.speakers[key],
                               self.confidences[key])
        else:
            raise AttributeError("Index must be either an integer or a slice object.")

    def __add__(self, other):
        assert type(other) is ExamplesSet
        return ExamplesSet(self.responses + other.responses,
                           self.prompts + other.prompts,
                           self.grades + other.grades,
                           self.speakers + other.speakers,
                           self.confidences + other.confidences)

    def __len__(self):
        return self.size

    def save_to(self, destination_dir):
        print("Saving set to:", destination_dir)
        save_set(destination_dir, responses='\n'.join(self.responses), confidences='\n'.join(self.confidences), prompts='\n'.join(self.prompts),
                 grades='\n'.join(self.grades), speakers='\n'.join(self.speakers))
        return


def save_set(destination_dir, responses, confidences, prompts, grades, speakers):
    if not os.path.isdir(destination_dir):
	    os.makedirs(destination_dir)

    with open(os.path.join(destination_dir, 'responses.txt'), 'w') as r:
        r.write(responses)
    with open(os.path.join(destination_dir, 'confidences.txt'), 'w') as c:
        c.write(confidences)
    with open(os.path.join(destination_dir, 'prompts.txt'), 'w') as p:
        p.write(prompts)
    with open(os.path.join(destination_dir, 'grades.txt'), 'w') as g:
        g.write(grades)
    with open(os.path.join(destination_dir, 'speakers.txt'), 'w') as s:
        s.write(speakers)
    return


def combine_divs(divisions, combine_idx):
    assert type(combine_idx) is list

    res = divisions[combine_idx[0]]
    for i in range(1, len(combine_idx)):
        res += divisions[combine_idx[i]]
    return res


def get_prompt_distribution(prompts):
    """
    Get the emprirical distribution of the prompts in the data
    :param prompts: list of str prompts
    :return: dict
    """
    prompt_freq = {}
    for prompt in prompts:
        prompt_freq.setdefault(prompt, 0)
        prompt_freq[prompt] += 1
    return prompt_freq


def evenly_divide_distribution(distrib_dict, k):
    """
    Given a dictionary of the form {key: count}, divide dict into k groups with approximately equal sum of counts
    in each. This function should efficiently find a sudo-optimal solution.
    :param distrib_dict: dict
    :return: list of k lists of keys, np.array(with sum of counts of each division
    """
    keys, counts = np.array(list(distrib_dict.keys())), np.array(list(distrib_dict.values()))
    # Sort in order of decreasing counts
    sort_idx = np.flip(np.argsort(counts), axis=0)
    keys = keys[sort_idx]
    counts = counts[sort_idx]

    # Allocate the prompts using this sudo-optimal algorithm
    division_counts = np.zeros([k], dtype=np.int32)
    key_divisions = [[] for i in range(k)]  # list of k empty lists
    for key, count in zip(keys, counts):
        # Get the division with the smallest count currently:
        div_idx = np.argmin(division_counts)
        # Add the key to that division
        division_counts[div_idx] += count
        key_divisions[div_idx].append(key)
    return key_divisions, division_counts


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

    # Get the prompt distribution in the data:
    prompt_dist_dict = get_prompt_distribution(prompts)

    # Split the prompts into k divisions with approx. equal number of examples each:
    prompt_divisions, div_counts = evenly_divide_distribution(prompt_dist_dict, args.k)  # todo: add k to args flags

    # Get percentage of data in each division:
    div_percentages = div_counts.astype(np.float32) / np.sum(div_counts).astype(np.float32)

    np.set_printoptions(precision=3, suppress=True)
    print("The target percentage for each division is: {} %.\n"
          "The actual percentage of examples in each division is:\n{}\n"
          "The example counts in each division are:\n{}\n".format(100.0 / float(args.k),
                                                                              div_percentages * 100.0,
                                                                  div_counts))

    # Convert the list of lists of prompts into a dict mapping to a division
    prompt_to_div_mapping = {}
    for i in range(args.k):
        for prompt in prompt_divisions[i]:
            prompt_to_div_mapping[prompt] = i

    # Create list of empty lists to store examples in each division
    responses_divided, confs_divided, prompts_divided, grades_divided, speakers_divided = ([[] for i in range(args.k)] for j in range(5))

    # Actually split all the examples into those divisions:
    for response, conf, prompt, grade, spkr in zip(responses, confs, prompts, grades, speakers):
        which_div = prompt_to_div_mapping[prompt]

        responses_divided[which_div].append(response)
        confs_divided[which_div].append(conf)
        prompts_divided[which_div].append(prompt)
        grades_divided[which_div].append(grade)
        speakers_divided[which_div].append(spkr)

    # Convert the divisions into sets because easier to handle
    example_set_divs = list(map(lambda x: ExamplesSet(*x),
                           zip(responses_divided, prompts_divided, grades_divided, speakers_divided, confs_divided)))

    # Now combine each of the kfolds and save them to a separate directory:
    for hold_out_idx in range(args.k):
        # Hold out one division and merge the others
        held_out_div = example_set_divs[hold_out_idx]

        remaining_idxs = set(range(args.k)) - set([hold_out_idx])
        remaining_divs = combine_divs(divisions=example_set_divs, combine_idx=list(remaining_idxs))


        held_out_div.save_to(os.path.join(args.destination_dir, 'dataset_' + str(hold_out_idx + 1), 'held_out'))
        remaining_divs.save_to(os.path.join(args.destination_dir, 'dataset_' + str(hold_out_idx + 1), 'main'))


    # Make a metadata file with info about the dataset:
    print("Writing the meta file.")
    meta_str = ""
    for i in range(args.k):
        meta_str += 'Split set {}:\n' \
                    '\tExamples held out: {}\n' \
                    '\tPercentage held out: {:6.3f} %\n'.format(i + 1, len(example_set_divs[i]),
                                                              div_percentages[i] * 100.0)
        with open(os.path.join(args.destination_dir, 'meta_info.txt'), 'w') as meta_file:
            meta_file.write(meta_str)

    print('Finished')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
