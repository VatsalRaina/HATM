#! /usr/bin/env python

import context

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

import core.utilities.tfrecord_utils as tfrecord_utils
from core.utilities.utilities import load_text

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('input_data_path', type=str,
                               help='absolute path to response data')
commandLineParser.add_argument('input_prompt_path', type=str,
                               help='absolute path to prompt data')
commandLineParser.add_argument('input_grade_path', type=str,
                               help='absolute path to ordered list of unique prompts')
commandLineParser.add_argument('input_spkr_path', type=str,
                               help='absolute path to ordered list of unique prompts')
commandLineParser.add_argument('input_wlist_path', type=str,
                               help='absolute path to input word list')
commandLineParser.add_argument('destination_dir', type=str,
                               help='absolute path location where to setup ')
commandLineParser.add_argument('--valid_fraction', type=float, default=0.1,
                               help='fraction of full data to reserve for validation')
commandLineParser.add_argument('--strip_start_end', action='store_true')

def main(argv=None):
    """Converts a dataset to tfrecords."""
    args = commandLineParser.parse_args()

    if os.path.isdir(args.destination_dir):
        print 'destination directory exists. Exiting...'
    else:
        os.makedirs(args.destination_dir)

    if not os.path.isdir('CMDs'):
        os.makedirs('CMDs')

    with open('CMDs/step_preprocess_data.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

        # Load responses and prompts as sequences of word ids
    responses, _ = load_text(args.input_data_path, args.input_wlist_path, strip_start_end=args.strip_start_end)
    prompts, _ = load_text(args.input_prompt_path, args.input_wlist_path, strip_start_end=args.strip_start_end)

    # Load up the prompts as sequences of words
    with open(args.input_prompt_path, 'r') as file:
        topics = [line.replace('\n', '') for line in file.readlines()]

    # Get unique set of topics and topic counts (and sort them)
    unique_topics, topic_counts = np.unique(topics, return_counts=True)
    topics = unique_topics[np.flip(np.argsort(topic_counts), 0)]
    topic_counts = np.flip(np.sort(topic_counts), 0)

    # Create dictionary for topics mapping sentence to topic id
    # Also create file of sorted topics and unigrams file
    # Unigram file later used for training
    topic_dict = {}
    with open(os.path.join(args.destination_dir,'unigrams.txt'), 'w') as ufile:
        with open(os.path.join(args.destination_dir,'sorted_topics.txt'), 'w') as tfile:
            for i, topic, count in zip(xrange(topics.shape[0]), topics, topic_counts):
                topic_dict[topic] = i
                ufile.write(str(i) + ',' + str(int(count)) + '\n')
                tfile.write(topic + '\n')

    # Load up the speakers and speakers
    grades = np.loadtxt(args.input_grade_path)
    with open(args.input_spkr_path, 'r') as file:
        speakers = np.asarray([line.replace('\n', '') for line in file.readlines()])

    # Create a list of topic IDs for every response
    with open(args.input_prompt_path, 'r') as file:
        q_ids = np.asarray([topic_dict[line.replace('\n', '')] for line in file.readlines()])

    ### Split data into train and validation  data sets
    n = len(responses)
    train_size = int(n * (1.0-args.valid_fraction))
    valid_size = n - train_size

    print 'Total dataset size', n, 'Train dataset size', train_size, 'Valid dataset size', valid_size

    np.random.seed(1000)

    permutation=np.random.choice(np.arange(n), n, replace=False)
    index_train=permutation[:train_size]
    inded_valid=permutation[train_size:]


    trn_responses = responses[index_train]
    trn_prompts = prompts[index_train]
    trn_q_ids = q_ids[index_train]
    trn_speakers = speakers[index_train]
    trn_grades = grades[index_train]

    valid_responses = responses[inded_valid]
    valid_prompts = prompts[inded_valid]
    valid_q_ids = q_ids[inded_valid]
    valid_speakers = speakers[inded_valid]
    valid_grades = grades[inded_valid]

    # Create the training TF Record file
    filename = 'relevance.train.tfrecords'
    print 'Writing', filename
    writer = tf.python_io.TFRecordWriter(os.path.join(args.destination_dir,filename))
    for response, prompt, q_id, grd, spkr in zip(trn_responses, trn_prompts, trn_q_ids, trn_grades, trn_speakers):
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                'targets': tfrecord_utils.float_feature([1.0]),
                'grade': tfrecord_utils.float_feature([grd]),
                'spkr': tfrecord_utils.bytes_feature([spkr]),
                'q_id': tfrecord_utils.int64_feature([q_id])
            }),
            feature_lists=tf.train.FeatureLists(feature_list={
                'response': tfrecord_utils.int64_feature_list(response),
                'prompt': tfrecord_utils.int64_feature_list(prompt)}))
        writer.write(example.SerializeToString())
    writer.close()

    # Create the validation TF Record file
    filename = 'relevance.valid.tfrecords'
    print 'Writing', filename
    writer = tf.python_io.TFRecordWriter(os.path.join(args.destination_dir,filename))
    for response, prompt, q_id, grd, spkr in zip(valid_responses, valid_prompts, valid_q_ids, valid_grades,
                                                 valid_speakers):
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                'targets': tfrecord_utils.float_feature([1.0]),
                'grade': tfrecord_utils.float_feature([grd]),
                'spkr': tfrecord_utils.bytes_feature([spkr]),
                'q_id': tfrecord_utils.int64_feature([q_id])
            }),
            feature_lists=tf.train.FeatureLists(feature_list={
                'response': tfrecord_utils.int64_feature_list(response),
                'prompt': tfrecord_utils.int64_feature_list(prompt)}))
        writer.write(example.SerializeToString())
    writer.close()



if __name__ == '__main__':
    main()
