from __future__ import print_function
import os
import time

import matplotlib
import numpy as np
import math

matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score as roc
import scipy
from scipy.special import loggamma

import tensorflow as tf
import tensorflow.contrib.slim as slim

import context
from core.basemodel import BaseModel
import core.utilities.utilities as util


class Importance(object):
    def __init__(self, hidden_layers, weights):

class AttentionTopicModel(BaseModel):
    def __init__(self, network_architecture=None, name=None, save_path='./', load_path=None, debug_mode=0, seed=100,
                 epoch=None):

        BaseModel.__init__(self, network_architecture=network_architecture, seed=seed, name=name, save_path=save_path,
                           load_path=load_path, debug_mode=debug_mode)

        with self._graph.as_default():
            with tf.variable_scope('input') as scope:
                self._input_scope = scope
                self.x_a = tf.placeholder(tf.int32, [None, None])
                self.x_q = tf.placeholder(tf.int32, [None, None])
                self.qlens = tf.placeholder(tf.int32, [None])
                self.alens = tf.placeholder(tf.int32, [None])
                self.y = tf.placeholder(tf.float32, [None, 1])
                self.maxlen = tf.placeholder(dtype=tf.int32, shape=[])

                self.dropout = tf.placeholder(tf.float32, [])
                self.batch_size = tf.placeholder(tf.int32, [])

            with tf.variable_scope('atm') as scope:
                self._model_scope = scope
                self._predictions, \
                self._probabilities, \
                self._logits, \
                self.attention, = self._construct_network(a_input=self.x_a,
                                                          a_seqlens=self.alens,
                                                          q_input=self.x_q,
                                                          q_seqlens=self.qlens,
                                                          n_samples=0,
                                                          maxlen=self.maxlen,
                                                          batch_size=self.batch_size,
                                                          keep_prob=self.dropout)

            self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        if load_path == None:
            with self._graph.as_default():
                init = tf.global_variables_initializer()
                self.sess.run(init)

                # If necessary, restore model from previous
        elif load_path != None:
            self.load(load_path=load_path, step=epoch)

    def _construct_network_ang_get_internal(self, a_input, a_seqlens, n_samples, q_input, q_seqlens, maxlen, batch_size,
                                            keep_prob=1.0):
        """

        :param a_input:
        :param a_seqlens:
        :param n_samples: Number of samples - used to repeat the response encoder output for the resampled prompt
        examples
        :param q_input:
        :param q_seqlens:
        :param maxlen:
        :param batch_size: The batch size before sampling!
        :param keep_prob:
        :return: predictions, probabilities, logits, attention
        """

        L2 = self.network_architecture['L2']
        initializer = self.network_architecture['initializer']

        # Question Encoder RNN
        with tf.variable_scope('Embeddings', initializer=initializer(self._seed)) as scope:
            embedding = slim.model_variable('word_embedding',
                                            shape=[self.network_architecture['n_in'],
                                                   self.network_architecture['n_ehid']],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            regularizer=slim.l2_regularizer(L2),
                                            device='/GPU:0')
            a_inputs = tf.nn.dropout(tf.nn.embedding_lookup(embedding, a_input, name='embedded_data'),
                                     keep_prob=keep_prob, seed=self._seed + 1)
            q_inputs = tf.nn.dropout(tf.nn.embedding_lookup(embedding, q_input, name='embedded_data'),
                                     keep_prob=keep_prob, seed=self._seed + 2)

            q_inputs_fw = tf.transpose(q_inputs, [1, 0, 2])
            q_inputs_bw = tf.transpose(tf.reverse_sequence(q_inputs, seq_lengths=q_seqlens, seq_axis=1, batch_axis=0),
                                       [1, 0, 2])

            a_inputs_fw = tf.transpose(a_inputs, [1, 0, 2])
            a_inputs_bw = tf.transpose(tf.reverse_sequence(a_inputs, seq_lengths=a_seqlens, seq_axis=1, batch_axis=0),
                                       [1, 0, 2])

        # Prompt Encoder RNN
        with tf.variable_scope('RNN_Q_FW', initializer=initializer(self._seed)) as scope:
            rnn_fw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            _, state_fw = rnn_fw(q_inputs_fw, sequence_length=q_seqlens, dtype=tf.float32)

        with tf.variable_scope('RNN_Q_BW', initializer=initializer(self._seed)) as scope:
            rnn_bw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            _, state_bw = rnn_bw(q_inputs_bw, sequence_length=q_seqlens, dtype=tf.float32)

        question_embeddings = tf.concat([state_fw[1], state_bw[1]], axis=1)
        question_embeddings = tf.nn.dropout(question_embeddings, keep_prob=keep_prob, seed=self._seed)

        # Response Encoder RNN
        with tf.variable_scope('RNN_A_FW', initializer=initializer(self._seed)) as scope:
            rnn_fw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            outputs_fw, _ = rnn_fw(a_inputs_fw, sequence_length=a_seqlens, dtype=tf.float32)

        with tf.variable_scope('RNN_A_BW', initializer=initializer(self._seed)) as scope:
            rnn_bw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            outputs_bw, _ = rnn_bw(a_inputs_bw, sequence_length=a_seqlens, dtype=tf.float32)

        outputs = tf.concat([outputs_fw, outputs_bw], axis=2)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.nn.dropout(outputs, keep_prob=keep_prob, seed=self._seed)

        a_seqlens = tf.tile(a_seqlens, [n_samples + 1])
        outputs = tf.tile(outputs, [1 + n_samples, 1, 1])

        hidden, attention = self._bahdanau_attention(memory=outputs, seq_lens=a_seqlens, maxlen=maxlen,
                                                     query=question_embeddings,
                                                     size=2 * self.network_architecture['n_rhid'],
                                                     batch_size=batch_size * (n_samples + 1))
        hidden_list = [hidden]

        with tf.variable_scope('Grader') as scope:
            for layer in xrange(self.network_architecture['n_flayers']):
                hidden = slim.fully_connected(hidden,
                                              self.network_architecture['n_fhid'],
                                              activation_fn=self.network_architecture['f_activation_fn'],
                                              weights_regularizer=slim.l2_regularizer(L2),
                                              scope="hidden_layer_" + str(layer))
                hidden_list.append(hidden)


            logits = slim.fully_connected(hidden,
                                          self.network_architecture['n_out'],
                                          activation_fn=None,
                                          scope="output_layer")
            probabilities = self.network_architecture['output_fn'](logits)
            predictions = tf.cast(tf.round(probabilities), dtype=tf.float32)

        return predictions, probabilities, logits, attention, hidden_list

    def _construct_network(self, a_input, a_seqlens, n_samples, q_input, q_seqlens, maxlen, batch_size, keep_prob=1.0):
        """

        :param a_input:
        :param a_seqlens:
        :param n_samples: Number of samples - used to repeat the response encoder output for the resampled prompt
        examples
        :param q_input:
        :param q_seqlens:
        :param maxlen:
        :param batch_size: The batch size before sampling!
        :param keep_prob:
        :return: predictions, probabilities, logits, attention
        """

        L2 = self.network_architecture['L2']
        initializer = self.network_architecture['initializer']

        # Question Encoder RNN
        with tf.variable_scope('Embeddings', initializer=initializer(self._seed)) as scope:
            embedding = slim.model_variable('word_embedding',
                                            shape=[self.network_architecture['n_in'],
                                                   self.network_architecture['n_ehid']],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            regularizer=slim.l2_regularizer(L2),
                                            device='/GPU:0')
            a_inputs = tf.nn.dropout(tf.nn.embedding_lookup(embedding, a_input, name='embedded_data'),
                                     keep_prob=keep_prob, seed=self._seed + 1)
            q_inputs = tf.nn.dropout(tf.nn.embedding_lookup(embedding, q_input, name='embedded_data'),
                                     keep_prob=keep_prob, seed=self._seed + 2)

            q_inputs_fw = tf.transpose(q_inputs, [1, 0, 2])
            q_inputs_bw = tf.transpose(tf.reverse_sequence(q_inputs, seq_lengths=q_seqlens, seq_axis=1, batch_axis=0),
                                       [1, 0, 2])

            a_inputs_fw = tf.transpose(a_inputs, [1, 0, 2])
            a_inputs_bw = tf.transpose(tf.reverse_sequence(a_inputs, seq_lengths=a_seqlens, seq_axis=1, batch_axis=0),
                                       [1, 0, 2])

        # Prompt Encoder RNN
        with tf.variable_scope('RNN_Q_FW', initializer=initializer(self._seed)) as scope:
            rnn_fw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            _, state_fw = rnn_fw(q_inputs_fw, sequence_length=q_seqlens, dtype=tf.float32)

        with tf.variable_scope('RNN_Q_BW', initializer=initializer(self._seed)) as scope:
            rnn_bw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            _, state_bw = rnn_bw(q_inputs_bw, sequence_length=q_seqlens, dtype=tf.float32)

        question_embeddings = tf.concat([state_fw[1], state_bw[1]], axis=1)
        question_embeddings = tf.nn.dropout(question_embeddings, keep_prob=keep_prob, seed=self._seed)

        # Response Encoder RNN
        with tf.variable_scope('RNN_A_FW', initializer=initializer(self._seed)) as scope:
            rnn_fw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            outputs_fw, _ = rnn_fw(a_inputs_fw, sequence_length=a_seqlens, dtype=tf.float32)

        with tf.variable_scope('RNN_A_BW', initializer=initializer(self._seed)) as scope:
            rnn_bw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            outputs_bw, _ = rnn_bw(a_inputs_bw, sequence_length=a_seqlens, dtype=tf.float32)

        outputs = tf.concat([outputs_fw, outputs_bw], axis=2)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.nn.dropout(outputs, keep_prob=keep_prob, seed=self._seed)

        a_seqlens = tf.tile(a_seqlens, [n_samples + 1])
        outputs = tf.tile(outputs, [1 + n_samples, 1, 1])

        hidden, attention = self._bahdanau_attention(memory=outputs, seq_lens=a_seqlens, maxlen=maxlen,
                                                     query=question_embeddings,
                                                     size=2 * self.network_architecture['n_rhid'],
                                                     batch_size=batch_size * (n_samples + 1))

        with tf.variable_scope('Grader') as scope:
            for layer in xrange(self.network_architecture['n_flayers']):
                hidden = slim.fully_connected(hidden,
                                              self.network_architecture['n_fhid'],
                                              activation_fn=self.network_architecture['f_activation_fn'],
                                              weights_regularizer=slim.l2_regularizer(L2),
                                              scope="hidden_layer_" + str(layer))
                hidden = tf.nn.dropout(hidden, keep_prob=keep_prob, seed=self._seed + layer)

            logits = slim.fully_connected(hidden,
                                          self.network_architecture['n_out'],
                                          activation_fn=None,
                                          scope="output_layer")
            probabilities = self.network_architecture['output_fn'](logits)
            predictions = tf.cast(tf.round(probabilities), dtype=tf.float32)

        return predictions, probabilities, logits, attention

    def _construct_prompt_encoder(self, p_input, p_seqlens):
        """ Construct RNNLM network
        Args:
          ?
        Returns:
          predictions, probabilities, logits, attention
        """

        L2 = self.network_architecture['L2']
        initializer = self.network_architecture['initializer']

        # Question Encoder RNN
        with tf.variable_scope('Embeddings', initializer=initializer(self._seed)) as scope:
            embedding = slim.model_variable('word_embedding',
                                            shape=[self.network_architecture['n_in'],
                                                   self.network_architecture['n_ehid']],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            regularizer=slim.l2_regularizer(L2),
                                            device='/GPU:0')

            p_inputs = tf.nn.embedding_lookup(embedding, p_input, name='embedded_data')

            p_inputs_fw = tf.transpose(p_inputs, [1, 0, 2])
            p_inputs_bw = tf.transpose(tf.reverse_sequence(p_inputs, seq_lengths=p_seqlens, seq_axis=1, batch_axis=0),
                                       [1, 0, 2])

        # Prompt Encoder RNN
        with tf.variable_scope('RNN_Q_FW', initializer=initializer(self._seed)) as scope:
            rnn_fw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            _, state_fw = rnn_fw(p_inputs_fw, sequence_length=p_seqlens, dtype=tf.float32)

        with tf.variable_scope('RNN_Q_BW', initializer=initializer(self._seed)) as scope:
            rnn_bw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            _, state_bw = rnn_bw(p_inputs_bw, sequence_length=p_seqlens, dtype=tf.float32)

            prompt_embeddings = tf.concat([state_fw[1], state_bw[1]], axis=1)

        return prompt_embeddings


    def predict_with_bruno_uncertainty(self, test_pattern, batch_size=20, cache_inputs=False, apply_bucketing=True):
        """
        Predict and save the uncertainty generated by bruno method
        :param test_pattern: filepath to dataset to run inference/evaluation on
        :param batch_size: int
        :param cache_inputs: Whether to save the response, prompts, response lengths, and prompt lengths in
        text form together with the predictions. Useful, since bucketing changes the order of the files and this allows
        to investigate which prediction corresponds to which prompt/response pair
        :param apply_bucketing: bool, whether to apply bucketing, i.e. group examples by their response length to
        minimise the overhead associated with zero-padding. If False, the examples will be evaluated in the original
        order as read from the file.

        :return: Depends on whether the inputs are being cached. If cache_inputs=False:
        returns test_loss, test_probabilities_array, test_true_labels_array
        If cache_inputs=True:
        returns test_loss, test_probabilities_array, test_true_labels_array, test_response_lengths,
                test_prompt_lengths, test_responses_list, test_prompts_list
        """
        with self._graph.as_default():
            test_files = tf.gfile.Glob(test_pattern)
            if apply_bucketing:
                batching_function = self._batch_func
            else:
                batching_function = self._batch_func_without_bucket
            test_iterator = self._construct_dataset_from_tfrecord(test_files,
                                                                  self._parse_func,
                                                                  self._map_func,
                                                                  batching_function,
                                                                  batch_size=batch_size,
                                                                  train=False,
                                                                  capacity_mul=100,
                                                                  num_threads=1)
            test_targets, \
            test_q_ids, \
            test_responses, \
            test_response_lengths, test_prompts, test_prompt_lens = test_iterator.get_next(name='valid_data')

            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                test_predictions, \
                test_probabilities, \
                test_logits, \
                test_attention = self._construct_network(a_input=test_responses,
                                                         a_seqlens=test_response_lengths,
                                                         n_samples=0,
                                                         q_input=test_prompts,
                                                         q_seqlens=test_prompt_lens,
                                                         maxlen=tf.reduce_max(test_response_lengths),
                                                         batch_size=batch_size,
                                                         keep_prob=1.0)

            loss = self._construct_xent_cost(targets=test_targets, logits=tf.squeeze(test_logits), pos_weight=1.0,
                                             is_training=False)
            self.sess.run(test_iterator.initializer)
            if cache_inputs:
                return self._predict_loop_with_caching(loss, test_probabilities, test_targets,
                                                       test_responses, test_response_lengths, test_prompts,
                                                       test_prompt_lens)
            else:
                return self._predict_loop(loss, test_probabilities, test_targets)

    def _predict_loop_with_caching(self, loss, test_probabilities, test_targets, test_responses, test_response_lengths,
                                   test_prompts, test_prompt_lens):
        test_loss = 0.0
        total_size = 0
        count = 0

        # Variables for storing the batch_ordered data
        test_responses_list = []
        test_prompts_list = []
        while True:
            try:
                batch_eval_loss, \
                batch_test_probs, \
                batch_test_targets, \
                batch_responses, \
                batch_response_lengths, \
                batch_prompts, \
                batch_prompt_lens = self.sess.run([loss,
                                                   test_probabilities,
                                                   test_targets,
                                                   test_responses,
                                                   test_response_lengths,
                                                   test_prompts,
                                                   test_prompt_lens])

                size = batch_test_probs.shape[0]
                test_loss += float(size) * batch_eval_loss
                if count == 0:
                    test_probs_arr = batch_test_probs  # shape: (num_batches, 1)
                    test_labels_arr = batch_test_targets[:, np.newaxis]  # becomes shape: (num_batches, 1)
                    test_response_lens_arr = batch_response_lengths[:, np.newaxis]  # becomes shape: (num_batches, 1)
                    test_prompt_lens_arr = batch_prompt_lens[:, np.newaxis]  # becomes shape: (num_batches, 1)
                else:
                    test_probs_arr = np.concatenate((test_probs_arr, batch_test_probs), axis=0)
                    test_labels_arr = np.concatenate((test_labels_arr, batch_test_targets[:, np.newaxis]), axis=0)
                    test_response_lens_arr = np.concatenate(
                        (test_response_lens_arr, batch_response_lengths[:, np.newaxis]), axis=0)
                    test_prompt_lens_arr = np.concatenate((test_prompt_lens_arr, batch_prompt_lens[:, np.newaxis]),
                                                          axis=0)
                test_responses_list.extend(list(batch_responses))  # List of numpy arrays!
                test_prompts_list.extend(list(batch_prompts))  # List of numpy arrays!

                total_size += size
                count += 1
            except:  # todo: tf.errors.OutOfRangeError:
                break

        test_loss = test_loss / float(total_size)

        return (test_loss,
                test_probs_arr,
                test_labels_arr.astype(np.int32),
                test_response_lens_arr.astype(np.int32),
                test_prompt_lens_arr.astype(np.int32),
                test_responses_list,
                test_prompts_list)

    def _predict_loop(self, loss, test_probabilities, test_targets):
        test_loss = 0.0
        total_size = 0
        count = 0

        # Variables for storing the batch_ordered data
        while True:
            try:
                batch_eval_loss, \
                batch_test_probs, \
                batch_test_targets = self.sess.run([loss,
                                                    test_probabilities,
                                                    test_targets])

                size = batch_test_probs.shape[0]
                test_loss += float(size) * batch_eval_loss
                if count == 0:
                    test_probs_arr = batch_test_probs  # shape: (num_batches, 1)
                    test_labels_arr = batch_test_targets[:, np.newaxis]  # becomes shape: (num_batches, 1)
                else:
                    test_probs_arr = np.concatenate((test_probs_arr, batch_test_probs), axis=0)
                    test_labels_arr = np.concatenate((test_labels_arr, batch_test_targets[:, np.newaxis]), axis=0)

                total_size += size
                count += 1
            except:  # todo: tf.errors.OutOfRangeError:
                break

        test_loss = test_loss / float(total_size)

        return (test_loss,
                test_probs_arr,
                test_labels_arr.astype(np.int32))

    def get_prompt_embeddings(self, prompts, prompt_lens, save_path):
        with self._graph.as_default():
            prompts = tf.convert_to_tensor(prompts, dtype=tf.int32)
            prompt_lens = tf.convert_to_tensor(prompt_lens, dtype=tf.int32)

            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                prompt_embeddings = self._construct_prompt_encoder(p_input=prompts, p_seqlens=prompt_lens)

            embeddings = self.sess.run(prompt_embeddings)

            path = os.path.join(save_path, 'prompt_embeddings.txt')
            np.savetxt(path, embeddings)


    def estimate_base_bruno_uncertainty(self, test_pattern, batch_size=20, cache_inputs=False, apply_bucketing=True):
        """
        :param test_pattern: filepath to dataset to run inference/evaluation on
        :param batch_size: int
        :param cache_inputs: Whether to save the response, prompts, response lengths, and prompt lengths in
        text form together with the predictions. Useful, since bucketing changes the order of the files and this allows
        to investigate which prediction corresponds to which prompt/response pair
        :param apply_bucketing: bool, whether to apply bucketing, i.e. group examples by their response length to
        minimise the overhead associated with zero-padding. If False, the examples will be evaluated in the original
        order as read from the file.

        :return: Depends on whether the inputs are being cached. If cache_inputs=False:
        returns test_loss, test_probabilities_array, test_true_labels_array
        If cache_inputs=True:
        returns test_loss, test_probabilities_array, test_true_labels_array, test_response_lengths,
                test_prompt_lengths, test_responses_list, test_prompts_list
        """
        with self._graph.as_default():
            test_files = tf.gfile.Glob(test_pattern)
            if apply_bucketing:
                batching_function = self._batch_func
            else:
                batching_function = self._batch_func_without_bucket
            test_iterator = self._construct_dataset_from_tfrecord(test_files,
                                                                  self._parse_func,
                                                                  self._map_func,
                                                                  batching_function,
                                                                  batch_size=batch_size,
                                                                  train=False,
                                                                  capacity_mul=100,
                                                                  num_threads=1)
            test_targets, \
            test_q_ids, \
            test_responses, \
            test_response_lengths, test_prompts, test_prompt_lens = test_iterator.get_next(name='valid_data')

            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                test_predictions, \
                test_probabilities, \
                test_logits, \
                test_attention = self._construct_network_and_get_internal(a_input=test_responses,
                                                                          a_seqlens=test_response_lengths,
                                                                          n_samples=0,
                                                                          q_input=test_prompts,
                                                                          q_seqlens=test_prompt_lens,
                                                                          maxlen=tf.reduce_max(test_response_lengths),
                                                                          batch_size=batch_size,
                                                                          keep_prob=1.0)

            loss = self._construct_xent_cost(targets=test_targets, logits=tf.squeeze(test_logits), pos_weight=1.0,
                                             is_training=False)
            self.sess.run(test_iterator.initializer)
            if cache_inputs:
                return self._predict_loop_with_caching(loss, test_probabilities, test_targets,
                                                       test_responses, test_response_lengths, test_prompts,
                                                       test_prompt_lens)
            else:
                return self._predict_loop(loss, test_probabilities, test_targets)
