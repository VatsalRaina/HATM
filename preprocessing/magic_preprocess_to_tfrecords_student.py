#! /usr/bin/env python

"""
Converts the processed data into a tfrecords dataset format.

----

Generates files:
relevance.train.tfrecords relevance.valid.tfrecords     (if --preprocessing_type = train)
relevance.test.tfrecords                                (if --preprocessing_type = test)
sorted-topics.txt unigrams.txt                          (if --sorted_topics_path left unspecified)
"""

from __future__ import print_function, division
import context
import argparse
import os
import sys
import shutil

import numpy as np
import tensorflow as tf

import core.utilities.tfrecord_utils as tfrecord_utils
from core.utilities.utilities import load_text

parser = argparse.ArgumentParser(description='Convert the processed data into a tfrecords dataset format')
parser.add_argument('data_dir', type=str,
                    help='absolute path to the directory with the processed responses, prompts, '
                         'speakers, grades etc. .txt data')
parser.add_argument('input_wlist_path', type=str,
                    help='absolute path to input word list')
parser.add_argument('destination_dir', type=str,
                    help='absolute path to directory location where to setup and save the tfrecords data')
parser.add_argument('--valid_fraction', type=float, default=0.1,
                    help='fraction of full data to reserve for validation')
parser.add_argument('--rand_seed', type=float, default=1000,
                    help='random seed to use when shuffling the data into validation and training sets')
parser.add_argument('--preprocessing_type', type=str, choices=['train', 'test'], default='train')
parser.add_argument('--sorted_topics_path', type=str, default='',
                    help='Absolute path to file with sorted topics as used for model training. Leave unspecified'
                         'if new mapping is to be generated.')
parser.add_argument('--responses_file', type=str, default='responses.txt')
parser.add_argument('--prompts_file', type=str, default='prompts.txt')
parser.add_argument('--grades_file', type=str, default='grades.txt')
parser.add_argument('--speakers_file', type=str, default='speakers.txt')
parser.add_argument('--targets_file', type=str, default='targets.txt')
parser.add_argument('--predictions_file', type=str, default='teacher_predictions.txt')
parser.add_argument('--remove_sentence_tags', action='store_true', help='whether to remove the <s> </s> tags at the '
                                                                        'beginning and end of each response/prompt')
parser.add_argument('--debug', action='store_true')


def write_to_tfrecords(filename, destination_dir, responses, prompts, q_ids, grades, speakers, targets, predictions,
                       debug=False):
    # Check that all the input lists are of equal lengths
    assert len({len(responses), len(prompts), len(q_ids), len(grades), len(speakers), len(targets), len(predictions)}) == 1

    # Create the training TF Record file
    print('Writing: ', filename)

    writer = tf.python_io.TFRecordWriter(os.path.join(destination_dir, filename))
    for response, prompt, q_id, grd, spkr, tgt, example_pred, idx in zip(responses, prompts, q_ids, grades, speakers, targets, predictions, range(len(q_ids))):
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                'targets': tfrecord_utils.float_feature([tgt]),
                'grade': tfrecord_utils.float_feature([float(grd)]),
                'teacher_pred': tfrecord_utils.float_feature(list(example_pred)),
                'spkr': tfrecord_utils.bytes_feature([spkr]),
                'q_id': tfrecord_utils.int64_feature([q_id]),
                'example_idx': tfrecord_utils.int64_feature([idx])  # Stores the example number for easy back-reference to txt files even when examples get shuffled (0 indexed)
            }),
            feature_lists=tf.train.FeatureLists(feature_list={
                'response': tfrecord_utils.int64_feature_list(response),
                'prompt': tfrecord_utils.int64_feature_list(prompt)}))
        if debug:
            # Print out the data that is going to be saved:
            print("-----------------\n", "EXAMPLE: \n", "Response: {}\nPrompt: {}\nQ_id: {}\n\ntarget: {}\ngrade: {}\n,teacher_pred: {}\nexample_num: {}\n\n".format(response, prompt, q_id, tgt, grd, example_pred, idx))
        writer.write(example.SerializeToString())
    writer.close()
    return


def generate_topic_dict(prompts_path, destination_dir):
    """Generate a topic dict mapping sentence to topic id and save files with the corresponding mappings in the destination directory"""
    # Load up the prompts as sequences of words
    with open(prompts_path, 'r') as file:
        topics = [line.replace('\n', '') for line in file.readlines()]

    # Get unique set of topics and topic counts
    unique_topics, topic_counts = np.unique(topics, return_counts=True)
    # Sort them in decreasing order
    topics = unique_topics[np.flip(np.argsort(topic_counts), 0)]
    topic_counts = np.flip(np.sort(topic_counts), 0)

    # Create dictionary for topics mapping sentence to topic id
    # Also create file of sorted topics and a unigrams file
    # Unigram file later used for training
    topic_dict = {}
    with open(os.path.join(destination_dir, 'unigrams.txt'), 'w') as ufile, open(
            os.path.join(destination_dir, 'sorted_topics.txt'), 'w') as tfile:
        for i, topic, count in zip(xrange(topics.shape[0]), topics, topic_counts):
            topic_dict[topic] = i
            ufile.write(str(i) + ',' + str(int(count)) + '\n')
            tfile.write(topic + '\n')
    return topic_dict


def load_topic_dict(sorted_topics_path):
    """Load a topic dict mapping sentence to topic id from a file generated previously for creating a training dataset."""
    topic_dict = {}
    i = 0
    # Load up sorted topics and (re)construct the topic dict so that I map each prompt word sequence to its q_id
    with open(os.path.join(args.sorted_topics_path), 'r') as tfile:
        for topic in tfile.readlines():
            topic_dict[topic.replace('\n', '')] = i
            i += 1
    return topic_dict


def main(args):
    """Converts a dataset to tfrecords."""

    if os.path.isdir(args.destination_dir):
        usr_response = raw_input("Destination directory exists. Type 'y' if you want to overwrite it: ")
        if usr_response != 'y':
            print('destination directory {} exists.\nExiting...'.format(args.destination_dir))
            exit()
    else:
        os.makedirs(args.destination_dir)

    shutil.copyfile(args.input_wlist_path, os.path.join(args.destination_dir, 'input.wlist.index'))

    # Get the paths to the relevant files
    responses_path = os.path.join(args.data_dir, args.responses_file)
    prompts_path = os.path.join(args.data_dir, args.prompts_file)
    grades_path = os.path.join(args.data_dir, args.grades_file)
    speakers_path = os.path.join(args.data_dir, args.speakers_file)
    predictions_path = os.path.join(args.data_dir, args.predictions_file)
    targets_path = os.path.join(args.data_dir, args.targets_file)

    required_files = [responses_path, prompts_path, grades_path, speakers_path, predictions_path, targets_path]

    # Assert the required files exist
    for path in required_files:
        if not os.path.isfile(path):
            print('File: {} doesn`t exist. Exiting...'.format(path))
            exit()

    # Cache the command:
    if not os.path.isdir(os.path.join(args.destination_dir, 'CMDs')):
        os.makedirs(os.path.join(args.destination_dir, 'CMDs'))
    with open(os.path.join(args.destination_dir, 'CMDs/preprocessing.cmd'), 'a') as cmd_cache:
        cmd_cache.write(' '.join(sys.argv) + '\n')
        cmd_cache.write('--------------------------------\n')

    # Load responses and prompts as sequences of word ids
    responses, _ = load_text(responses_path, args.input_wlist_path, strip_start_end=args.remove_sentence_tags)
    prompts, _ = load_text(prompts_path, args.input_wlist_path, strip_start_end=args.remove_sentence_tags)

    # Load up the speakers and grades
    with open(grades_path, 'r') as file:
        grades = np.asarray([line.replace('\n', '') for line in file.readlines()])
    with open(speakers_path, 'r') as file:
        speakers = np.asarray([line.replace('\n', '') for line in file.readlines()])
    # Load the teacher predictions
    predictions = np.loadtxt(predictions_path, dtype=np.float32)
    # load targets
    targets = np.loadtxt(targets_path, dtype=np.float32)

    # Create or load the topic ID dictionary:
    if args.sorted_topics_path == '':
        # Generate and save new topic ID mapping
        topic_dict = generate_topic_dict(prompts_path, args.destination_dir)
    else:
        # Load a pre-existing topic ID mapping
        topic_dict = load_topic_dict(args.sorted_topics_path)

    # Create a list of topic IDs for every response
    with open(prompts_path, 'r') as file:
        # Use id -1 if topic not in topic_dict (not present in training set)
        q_ids = np.asarray([topic_dict.get(line.replace('\n', ''), -1) for line in file.readlines()])

    if args.preprocessing_type == 'train':
        ### Split data into train and validation data sets
        n = len(responses)
        train_size = int(n * (1.0 - args.valid_fraction))
        valid_size = n - train_size

        print('Total dataset size', n, 'Train dataset size', train_size, 'Valid dataset size', valid_size)

        np.random.seed(args.rand_seed)

        permutation = np.random.choice(np.arange(n), n, replace=False)
        index_train = permutation[:train_size]
        index_valid = permutation[train_size:]

        trn_responses = responses[index_train]
        trn_prompts = prompts[index_train]
        trn_q_ids = q_ids[index_train]
        trn_speakers = speakers[index_train]
        trn_grades = grades[index_train]
        trn_targets = targets[index_train]
        trn_predictions = predictions[index_train]

        print("Number training examples: {}".format(len(trn_responses)))

        valid_responses = responses[index_valid]
        valid_prompts = prompts[index_valid]
        valid_q_ids = q_ids[index_valid]
        valid_speakers = speakers[index_valid]
        valid_grades = grades[index_valid]
        valid_targets = targets[index_valid]
        valid_predictions = predictions[index_valid]

        print("Number validation examples: {}".format(len(valid_responses)))

        # Create the training TF Record file
        write_to_tfrecords('relevance.train.tfrecords', args.destination_dir, trn_responses, trn_prompts, trn_q_ids,
                           trn_grades, trn_speakers, trn_targets, trn_predictions, debug=args.debug)

        # Create the validation TF Record file
        write_to_tfrecords('relevance.valid.tfrecords', args.destination_dir, valid_responses, valid_prompts,
                           valid_q_ids, valid_grades, valid_speakers, valid_targets, valid_predictions,
                           debug=args.debug)

        # Write a metadata file for convenience:
        with open(os.path.join(args.destination_dir, 'dataset_meta.txt'), 'w') as meta_file:
            meta_string = 'num_examples_train:\t{}\nnum_examples_valid:\t{}\nnum_unique_topics:\t{}'.format(
                len(trn_responses), len(valid_responses), len(topic_dict))
            meta_file.write(meta_string)

    elif args.preprocessing_type == 'test':
        write_to_tfrecords('relevance.test.tfrecords', args.destination_dir, responses, prompts, q_ids, grades,
                           speakers, targets, predictions, debug=args.debug)

        # Write a metadata file for convenience:
        with open(os.path.join(args.destination_dir, 'dataset_meta.txt'), 'w') as meta_file:
            meta_string = 'num_examples:\t{}\nnum_unique_topics:\t{}'.format(len(responses), len(topic_dict))
            meta_file.write(meta_string)

    print('Finished')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
