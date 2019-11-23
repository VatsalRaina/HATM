#! /usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys
try:
    import cPickle as pickle
except:
    import pickle

import context
from atm.atm import AttentionTopicModel
from core.utilities.utilities import activation_dict
from core.utilities.utilities import initializer_dict
from core.utilities.utilities import activation_fn_list
from core.utilities.utilities import initializer_list
from core.utilities.utilities import get_num_topics_from_meta

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('data_path', type=str,
                               help='absolute path to data')
commandLineParser.add_argument('destination_dir', type=str,
                               help='absolute path to directory location where to setup')
commandLineParser.add_argument('library_path', type=str,
                               help='absolute path to the repository with attention-topic-model code files')
commandLineParser.add_argument('wlist_path', type=str,
                               help='path to word list file (vocabulary for embeddings)')
commandLineParser.add_argument('--n_topics', type=int, default=379,
                               help='which should be loaded')
commandLineParser.add_argument('--meta_data_path', type=str, default=None, help="Path to metadata file with number of "
                                                                                "topics in the training data. This can"
                                                                                "be used to override the n_topics "
                                                                                "flag and load num. of topics from a"
                                                                                "file.")
commandLineParser.add_argument('--name', type=str, default='attention_based_topic_model',
                               help='Specify the name of the model')
commandLineParser.add_argument('--seed', type=int, default=100,
                               help='Specify the global random seed')
commandLineParser.add_argument('--debug', type=int, choices=[0, 1, 2], default=0,
                               help='Specify the debug output level')
commandLineParser.add_argument('--load_path', type=str, default=None,
                               help='Specify path to model which should be loaded ')
commandLineParser.add_argument('--save_path', type=str, default='./',
                               help='Specify path to model which should be loaded')
commandLineParser.add_argument('--init', type=str, default=None,
                               help='Specify path to from which to initialize model (to initialise embedding weights)')
commandLineParser.add_argument('--epoch', type=str, default=None,
                               help='which should be loaded')
# Model parameters
commandLineParser.add_argument('--n_in', type=int, default=62416)
commandLineParser.add_argument('--n_ehid', type=int, default=200,
                               help='which should be loaded')
commandLineParser.add_argument('--n_fhid', type=int, default=180,
                               help='which should be loaded')
commandLineParser.add_argument('--n_rhid', type=int, default=200,
                               help='which should be loaded')
commandLineParser.add_argument('--n_phid', type=int, default=200,
                               help='which should be loaded')
commandLineParser.add_argument('--n_rlayers', type=int, default=1,
                               help='which should be loaded')
commandLineParser.add_argument('--n_flayers', type=int, default=2,
                               help='which should be loaded')
commandLineParser.add_argument('--n_out', type=int, default=1,
                               help='which should be loaded')  # For a prior network, set to two!
commandLineParser.add_argument('--l2', type=float, default=4e-7,
                               help='which should be loaded')
commandLineParser.add_argument('--sample_distortion', type=bool, default=False,
                               help='which should be loaded')
commandLineParser.add_argument('--f_activation_fn',
                               choices=activation_fn_list,
                               default='lrelu',
                               help='which should be loaded')
commandLineParser.add_argument('--r_activation_fn', choices=activation_fn_list,
                                default='tanh',
                                help='which should be loaded')
commandLineParser.add_argument('--output_fn', choices=activation_fn_list,
                                default='sigmoid',
                                help='output activation function')  # For a prior network, set to softmax!
commandLineParser.add_argument('--initializer', choices=initializer_list,
                                default='xavier')


def main(argv=None):
    """Converts a dataset to tfrecords."""
    args = commandLineParser.parse_args()

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_setup_attention_grader.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    if os.path.isdir(args.destination_dir):
        print('destination directory exists. Exiting...')
    else:
        os.makedirs(args.destination_dir)

    # Link and and create directories
    os.chdir(args.destination_dir)
    os.mkdir('model')
    os.symlink(args.data_path, 'data')
    os.symlink(args.library_path, 'atm')


    if args.meta_data_path is None:
        n_topics = args.n_topics
    else:
        n_topics = get_num_topics_from_meta(args.meta_data_path)
    print("Num topics used in model: ", n_topics)

    #Define network architecture
    network_architecture = dict(model_name=args.name,  # Define Model Type
                                n_in=args.n_in,  # data input dimension
                                n_ehid=args.n_ehid,  # Size of word embeddings
                                n_fhid=args.n_fhid,  # feed-forward hidden layer size
                                n_rhid=args.n_rhid,  # Response hidden layer size (doubled for BRNN)
                                n_phid=args.n_phid,  # Prompt hidden layer size (doubled for BRNN)
                                n_rlayers=args.n_rlayers,  # Number of Recurrent Layers. Unused
                                n_flayers=args.n_flayers,  # Number of feed-forward hidden layers
                                n_out=args.n_out,  # output dimensionality
                                f_activation_fn=activation_dict[args.f_activation_fn],
                                # Activation Function feed-forward layers
                                r_activation_fn=activation_dict[args.r_activation_fn],
                                # Activation Function Recurrent Layers
                                output_fn=activation_dict[args.output_fn],  # Output function
                                initializer=initializer_dict[args.initializer],  # Parameter Initializer
                                sample_distortion=args.sample_distortion,
                                L2=args.l2,  # L2 weight decay
                                n_topics=n_topics  # Number of topics in training data
    )

    # initialize the model and save intialized parameters
    atm = AttentionTopicModel(network_architecture=network_architecture,
                              seed=args.seed,
                              name=args.name,
                              save_path=args.save_path,
                              load_path=args.load_path,
                              debug_mode=args.debug,
                              epoch=args.epoch)
    atm.save()

if __name__ == '__main__':
    main()
