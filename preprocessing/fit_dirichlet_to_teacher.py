#! /usr/bin/env python

"""
Fits parameters alpha to teacher predictions for each example as to maximise log-likelihood.

Creates a file:
alphas.txt

"""
from __future__ import print_function
import context

import os
import sys
import argparse

import numpy as np

import sys
import scipy
import scipy.optimize
import scipy.stats as stats
from core.utilities.utilities import nll_exp


parser = argparse.ArgumentParser(description='Create k new k-fold split datasets that can be used for generating '
                                             'teacher data on unseen prompts from a single dataset.')
parser.add_argument('data_dir', type=str,
                    help='absolute path to the directory with the processed responses,'
                         'prompts, speakers, grades etc. .txt data')
parser.add_argument('--teacher_predictions_filename', type=str, default='teacher_predictions.txt')
parser.add_argument('--output_filename', type=str, default='alphas.txt')
# parser.add_argument('--batch_size', type=int, default=1000, help='batch size for fitting the 2 alpha parameters')
# parser.add_argument('--learning_rate', type=float, default=0.04, help='learning rate for fitting the dirchlets')
# parser.add_argument('--num_steps', type=int, default=500, help='number of steps to fit the dirichlets')


# def nll_loss(alphas, teacher_predictions):
#     alpha1, alpha2 = tf.split(alphas, num_or_size_splits=2, axis=1)
#
#     log_likelihood_const_part = tf.lgamma(alpha1 + alpha2) - tf.lgamma(alpha1) - tf.lgamma(alpha2)
#     log_likelihood_var_part = tf.log(teacher_predictions) * (alpha1 - 1.0) + tf.log(1.0 - teacher_predictions) * (
#         alpha2 - 1.0)
#     log_likelihood = log_likelihood_const_part + log_likelihood_var_part
#
#     nll_loss = -1.0 * tf.reduce_mean(log_likelihood, axis=1)  # Take the mean over individual ensemble predictions
#     nll_cost = tf.reduce_mean(nll_loss)  # Take mean batch-wise (over the number of examples)
#     return nll_cost


def main(args):
    # Get the paths to the relevant files
    teacher_pred_path = os.path.join(args.data_dir, args.teacher_predictions_filename)

    # Assert the required files exist
    if not os.path.isfile(teacher_pred_path):
        print('File: {} doesn`t exist. Exiting...'.format(teacher_pred_path))
        exit()

    # Cache the command:
    if not os.path.isdir(os.path.join(args.data_dir, 'CMDs')):
        os.makedirs(os.path.join(args.data_dir, 'CMDs'))
    with open(os.path.join(args.data_dir, 'CMDs/preprocessing.cmd'), 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('-----------------------------------------------------------\n')
    print("Command cached")

    # Open the files
    teacher_predictions = np.loadtxt(teacher_pred_path, dtype=np.float32)
    print('Teacher predictions loaded')


    # Fit a Dirichlet to each example
    num_examples = teacher_predictions.shape[0]
    log_alphas = np.empty([teacher_predictions.shape[0], 2], dtype=np.float32)
    for i in xrange(teacher_predictions.shape[0]):
        row = teacher_predictions[i]
        log_alphas_row = scipy.optimize.fmin(lambda x: nll_exp(x, row), np.array([1., 1.]), disp=False)
        log_alphas[i] = log_alphas_row

        # Print every n steps
        if i % 100 == 0:
            print('Step {} out of {}'.format(i, num_examples))
    alphas = np.exp(log_alphas).astype(np.float32)

    # Save the results
    save_path = os.path.join(args.data_dir, args.output_filename)
    np.savetxt(save_path, alphas)
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

