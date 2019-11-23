#! /usr/bin/env python


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
commandLineParser.add_argument('input_tgt_path', type=str,
                               help='absolute path to input word list')
commandLineParser.add_argument('input_wlist_path', type=str,
                               help='absolute path to input word list')
commandLineParser.add_argument('sorted_topics_path', type=str,
                               help='absolute path to response data')
commandLineParser.add_argument('destination_dir', type=str,
                               help='absolute path location where to setup ')
commandLineParser.add_argument('name', type=str,
                               help='absolute path location where to setup ')


def main(argv=None):
    """Converts a dataset to tfrecords."""
    args = commandLineParser.parse_args()

    if not os.path.isdir(args.destination_dir):
        os.makedirs(args.destination_dir)

    if not os.path.isdir('CMDs'):
        os.makedirs('CMDs')

    with open('CMDs/step_preprocess_test_data.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Load responses and prompts as sequences of word ids
    responses, _ = load_text(args.input_data_path, args.input_wlist_path)
    prompts, _ = load_text(args.input_prompt_path, args.input_wlist_path)

    # Load up the grades, targets and speakers
    grades = np.loadtxt(args.input_grade_path)
    targets = np.loadtxt(args.input_tgt_path, dtype=np.float32)
    with open(args.input_spkr_path, 'r') as file:
        speakers = np.asarray([line.replace('\n', '') for line in file.readlines()])

    # Load up sorted topics and (re)construct the topic dict so that I map each prompt word sequence to its q_id
    topic_dict = {}
    i = 0
    with open(os.path.join(args.sorted_topics_path), 'r') as tfile:
        for topic in tfile.readlines():
            topic_dict[topic.replace('\n', '')] = i
            i += 1

    # Load up the prompts as sequences of words and convert to q_id
    with open(args.input_prompt_path, 'r') as file:
        q_ids = np.asarray([topic_dict.get(line.replace('\n', ''), default=-1) for line in file.readlines()])

    # Create the training TF Record file
    filename = args.name + '.tfrecords'
    print 'Writing', filename

    writer = tf.python_io.TFRecordWriter(os.path.join(args.destination_dir, filename))
    for response, prompt, q_id, grd, spkr, tgt in zip(responses, prompts, q_ids, grades, speakers, targets):
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                'targets': tfrecord_utils.float_feature([tgt]),
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
