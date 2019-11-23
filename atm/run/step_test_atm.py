#! /usr/bin/env python

import argparse
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as roc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
sns.set()

import context
from atm.atm import AttentionTopicModel
from core.utilities.utilities import text_to_array, IdToWordConverter

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('--load_path', type=str, default='./',
                               help='Specify path to model which should be loaded')
commandLineParser.add_argument('--debug', type=int, default=0,
                               help='Specify path to model which should be loaded')
commandLineParser.add_argument('--epoch', type=str, default=None,
                               help='which should be loaded')
commandLineParser.add_argument('--wlist_path', type=str, default=None)
commandLineParser.add_argument('--save_reordered_input', action='store_true', help="Whether to save the input words and"
                                                                                   "responses to model as reordered during"
                                                                                   "evaluation to the evaluation directory.")
commandLineParser.add_argument('--preserve_order', action='store_true', help="Whether to preserve order of the "
                                                                             "input files, so that outputted "
                                                                             "probabilities appear in the same order"
                                                                             "as prompt/response pairs in input. "
                                                                             "The disadvantage of using this flag is "
                                                                             "that bucketing won't be applied to the "
                                                                             "data, and the evaluation might take "
                                                                             "significantly longer to run.")
commandLineParser.add_argument('--make_plots', action='store_true',
                               help="Whether to make the ROC and PR plots for the model.")
commandLineParser.add_argument('data_pattern', type=str,
                               help='absolute path to response data')
commandLineParser.add_argument('output_dir', type=str, default=None,
                               help='which should be loaded')



def main(args):
    if args.save_reordered_input:
        assert args.wlist_path != None

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_test_attention_grader.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Initialize and Run the Model
    atm = AttentionTopicModel(network_architecture=None,
                              load_path=args.load_path,
                              debug_mode=args.debug,
                              epoch=args.epoch)

    apply_bucketing = (not args.preserve_order)

    start_time = time.time()

    if args.save_reordered_input:
        test_loss, \
        test_probs_arr, \
        test_labels_arr, \
        test_response_lens_arr, \
        test_prompt_lens_arr, \
        test_responses_list, \
        test_prompts_list = atm.predict(args.data_pattern, cache_inputs=True, apply_bucketing=apply_bucketing)
    else:
        test_loss, \
        test_probs_arr, \
        test_labels_arr = atm.predict(args.data_pattern, cache_inputs=False, apply_bucketing=apply_bucketing)

    end_time = time.time()
    print("Time taken for evaluating the dataset: {} minutes".format((end_time - start_time) / 60.0))

    # Save the numerical output data
    data = np.concatenate((test_labels_arr, test_probs_arr), axis=1)
    np.savetxt(os.path.join(args.output_dir, 'labels-probs.txt'), data)
    np.savetxt(os.path.join(args.output_dir, 'labels.txt'), test_labels_arr)
    np.savetxt(os.path.join(args.output_dir, 'predictions.txt'), test_probs_arr)

    if args.save_reordered_input:
        np.savetxt(os.path.join(args.output_dir, 'response_lengths.txt'), test_response_lens_arr)
        np.savetxt(os.path.join(args.output_dir, 'prompt_lengths.txt'), test_prompt_lens_arr)

        # Retrieve prompts and responses and convert to a list of lists
        responses_word_ids = map(lambda i: list(test_responses_list[i][:test_response_lens_arr[i, 0]]),
                                 range(len(test_responses_list)))
        prompts_word_ids = map(lambda i: list(test_prompts_list[i][:test_prompt_lens_arr[i, 0]]),
                               range(len(test_prompts_list)))
        # Save the prompts and responses
        with open(os.path.join(args.output_dir, 'responses_word_ids.txt'), 'w') as response_file:
            responses_str_lines = map(lambda x: ' '.join(map(lambda w_id: str(w_id), x)), responses_word_ids)
            response_file.write('\n'.join(responses_str_lines))
        with open(os.path.join(args.output_dir, 'prompts_word_ids.txt'), 'w') as prompt_file:
            prompts_str_lines = map(lambda x: ' '.join(map(lambda w_id: str(w_id), x)), prompts_word_ids)
            prompt_file.write('\n'.join(prompts_str_lines))

        # Process responses and prompts to get actual words
        if args.wlist_path is not None:
            id_converter = IdToWordConverter(args.wlist_path)
            responses_words = map(lambda id_list: ' '.join(id_converter.id_list_to_word(id_list)), responses_word_ids)
            prompts_words = map(lambda id_list: ' '.join(id_converter.id_list_to_word(id_list)), prompts_word_ids)
            with open(os.path.join(args.output_dir, 'responses.txt'), 'w') as response_file:
                response_file.write('\n'.join(responses_words))
            with open(os.path.join(args.output_dir, 'prompts.txt'), 'w') as prompt_file:
                prompt_file.write('\n'.join(prompts_words))

    # Do evaluations, calculate metrics, etc...
    roc_score = roc(np.squeeze(test_labels_arr), np.squeeze(test_probs_arr))

    test_labels_arr = np.squeeze(test_labels_arr)
    test_probs_arr = np.squeeze(test_probs_arr)

    precision_rel, recall_rel, thresholds = precision_recall_curve(test_labels_arr, test_probs_arr)
    aupr_rel = auc(recall_rel, precision_rel)

    precision_nonrel, recall_nonrel, thresholds = precision_recall_curve(1 - test_labels_arr, 1. - test_probs_arr)
    aupr_nonrel = auc(recall_nonrel, precision_nonrel)

    if args.make_plots:
        fpr, tpr, thresholds = roc_curve(np.asarray(test_labels_arr, dtype=np.int32), test_probs_arr)
        plt.plot(fpr, tpr, c='r')
        plt.plot([0, 1], [0, 1], 'k--', lw=4)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig(os.path.join(args.output_dir, 'test_roc_curve.png'))
        plt.close()

        plt.plot(recall_rel, precision_rel)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 1.0)
        plt.savefig(os.path.join(args.output_dir, 'test_pr_curve_relevant.png'))
        plt.close()

        plt.plot(recall_nonrel, precision_nonrel)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 1.0)
        plt.savefig(os.path.join(args.output_dir, 'test_pr_curve_off-topic.png'))
        plt.close()

    with open(os.path.join(args.output_dir, 'results.txt'), 'a') as f:
        f.write('Epoch: ' + str(args.epoch) + '\n')
        f.write('ROC AUC:' + str(np.round(roc_score, 3)) + '\n')
        f.write('ROC PR Detect Relevant:' + str(np.round(aupr_rel, 3)) + '\n')
        f.write('ROC PR Detect Non-Relevant:' + str(np.round(aupr_nonrel, 3)) + '\n')
        f.write('Cross Entropy:' + str(np.round(test_loss, 3)) + '\n\n')

if __name__ == '__main__':
    args = commandLineParser.parse_args()
    main(args)
