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

import context
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2

from core.basemodel import BaseModel
import core.utilities.utilities as util


class SimGrid(BaseModel):
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
                self.max_q_len = tf.placeholder(dtype=tf.int32, shape=[])
                self.max_a_len = tf.placeholder(dtype=tf.int32, shape=[])
                self.dropout = tf.placeholder(tf.float32, [])
                self.batch_size = tf.placeholder(tf.int32, [])
            
            with tf.variable_scope('atm') as scope:
                self._model_scope = scope
                self._predictions, \
                self._probabilities, \
                self._logits = self._construct_network(a_input=self.x_a,
                                                          a_seqlens=self.alens,
                                                          q_input=self.x_q,
                                                          q_seqlens=self.qlens,
                                                          max_q_len=self.max_q_len,
                                                          max_a_len=self.max_a_len,
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

    def cosine(self, xx, yy, ax):
        """
        Need to move to utilities python file
        """
        x_norm = tf.nn.l2_normalize(xx, dim=ax)
        y_norm = tf.nn.l2_normalize(yy, dim=ax)
        cos = tf.reduce_sum(x_norm * y_norm, axis=ax)
        cosx = tf.clip_by_value(cos, -1.0, 1.0)
        return cosx

    def euclidean(self, xx, yy, ax):
        diff = xx - yy
        square_diff = tf.square(diff)
        return tf.reduce_sum(square_diff, axis=ax)

    def _construct_network(self, a_input, a_seqlens, q_input, q_seqlens, max_q_len, max_a_len, batch_size, is_training=False, keep_prob=1.0):
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

        # Get IDFs for each content word
        np_idf = np.loadtxt("/home/alta/relevance/vr311/data_vatsal/BULATS/content_words/idf_global.txt", dtype=np.float32)
        idf = tf.convert_to_tensor(np_idf, dtype=tf.float32)

        L2 = self.network_architecture['L2']
        initializer = self.network_architecture['initializer']

        if is_training:
            batch_size *= 2

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


            embedding2 = slim.model_variable('word_embedding2',
                                            shape=[self.network_architecture['n_in'],
                                                   self.network_architecture['n_ehid']],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            regularizer=slim.l2_regularizer(L2),
                                            device='/GPU:0')

            a_inputs2 = tf.nn.dropout(tf.nn.embedding_lookup(embedding2, a_input, name='embedded_data'),
                                     keep_prob=keep_prob, seed=self._seed + 11)
            q_inputs2 = tf.nn.dropout(tf.nn.embedding_lookup(embedding2, q_input, name='embedded_data'),
                                     keep_prob=keep_prob, seed=self._seed + 22)


            embedding3 = slim.model_variable('word_embedding3',
                                            shape=[self.network_architecture['n_in'],
                                                   self.network_architecture['n_ehid']],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            regularizer=slim.l2_regularizer(L2),
                                            device='/GPU:0')

            a_inputs3 = tf.nn.dropout(tf.nn.embedding_lookup(embedding3, a_input, name='embedded_data'),
                                     keep_prob=keep_prob, seed=self._seed + 111)
            q_inputs3 = tf.nn.dropout(tf.nn.embedding_lookup(embedding3, q_input, name='embedded_data'),
                                     keep_prob=keep_prob, seed=self._seed + 222)

        """
            TEMP to try the impact of using BiLSTMs for learning separate word embeddings for prompt and response words
            
            q_inputs_fw = tf.transpose(q_inputs, [1, 0, 2])
            q_inputs_bw = tf.transpose(tf.reverse_sequence(q_inputs, seq_lengths=q_seqlens, seq_axis=1, batch_axis=0),
                                       [1, 0, 2])

            a_inputs_fw = tf.transpose(a_inputs, [1, 0, 2])
            a_inputs_bw = tf.transpose(tf.reverse_sequence(a_inputs, seq_lengths=a_seqlens, seq_axis=1, batch_axis=0),
                                       [1, 0, 2])

        # Response Encoder RNN
        with tf.variable_scope('RNN_Q_FW', initializer=initializer(self._seed)) as scope:
            rnn_fw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            outputs_fw, _ = rnn_fw(q_inputs_fw, sequence_length=q_seqlens, dtype=tf.float32)

        with tf.variable_scope('RNN_Q_BW', initializer=initializer(self._seed)) as scope:
            rnn_bw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            outputs_bw, _ = rnn_bw(q_inputs_bw, sequence_length=q_seqlens, dtype=tf.float32)

        outputs = tf.concat([outputs_fw, outputs_bw], axis=2)
        outputs = tf.transpose(outputs, [1, 0, 2])
        q_inputs = tf.nn.dropout(outputs, keep_prob=keep_prob, seed=self._seed)


        # Response Encoder RNN
        with tf.variable_scope('RNN_A_FW', initializer=initializer(self._seed)) as scope:
            rnn_fw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            outputs_fw, _ = rnn_fw(a_inputs_fw, sequence_length=a_seqlens, dtype=tf.float32)

        with tf.variable_scope('RNN_A_BW', initializer=initializer(self._seed)) as scope:
            rnn_bw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            outputs_bw, _ = rnn_bw(a_inputs_bw, sequence_length=a_seqlens, dtype=tf.float32)

        outputs = tf.concat([outputs_fw, outputs_bw], axis=2)
        outputs = tf.transpose(outputs, [1, 0, 2])
        a_inputs = tf.nn.dropout(outputs, keep_prob=keep_prob, seed=self._seed)
        """

        # Construct similarity grid of distances
        
        a_inputs = tf.tile(a_inputs, [1, max_q_len ,1])
        a_inputs = tf.transpose(tf.reshape(a_inputs, [batch_size,  max_q_len,  max_a_len, self.network_architecture['n_ehid']]), perm=[0,2,1,3])
        q_inputs = tf.tile(q_inputs, [1, max_a_len, 1])
        q_inputs = tf.reshape(q_inputs, [batch_size, max_a_len, max_q_len, self.network_architecture['n_ehid']])

        gridCos = self.cosine(a_inputs, q_inputs, ax=3)
        # Convert to 4D tensor [batch, height, width, channels] - for now, there is only one channel
        gridCos = tf.expand_dims(gridCos, axis=3, name='expanded_grid')

        """
        Try multiple channels where each channel is a cosine distance with different initialisation seeds
        Kind of like ensembling
        """
        a_inputs2 = tf.tile(a_inputs2, [1, max_q_len ,1])
        a_inputs2 = tf.transpose(tf.reshape(a_inputs2, [batch_size,  max_q_len,  max_a_len, self.network_architecture['n_ehid']]), perm=[0,2,1,3])
        q_inputs2 = tf.tile(q_inputs2, [1, max_a_len, 1])
        q_inputs2 = tf.reshape(q_inputs2, [batch_size, max_a_len, max_q_len, self.network_architecture['n_ehid']])

        gridCos2 = self.cosine(a_inputs2, q_inputs2, ax=3)
        # Convert to 4D tensor [batch, height, width, channels] - for now, there is only one channel
        gridCos2 = tf.expand_dims(gridCos2, axis=3, name='expanded_grid2')


        a_inputs3 = tf.tile(a_inputs3, [1, max_q_len ,1])
        a_inputs3 = tf.transpose(tf.reshape(a_inputs3, [batch_size,  max_q_len,  max_a_len, self.network_architecture['n_ehid']]), perm=[0,2,1,3])
        q_inputs3 = tf.tile(q_inputs3, [1, max_a_len, 1])
        q_inputs3 = tf.reshape(q_inputs3, [batch_size, max_a_len, max_q_len, self.network_architecture['n_ehid']])

        gridCos3 = self.cosine(a_inputs3, q_inputs3, ax=3)
        # Convert to 4D tensor [batch, height, width, channels] - for now, there is only one channel
        gridCos3 = tf.expand_dims(gridCos3, axis=3, name='expanded_grid3')


       # gridEuc = self.cosine(a_inputs, q_inputs, ax=3)
        # Convert to 4D tensor [batch, height, width, channels] - 2 channels now
       # gridEuc = tf.expand_dims(gridEuc, axis=3, name='expanded_grid')

        """        
        grid_p_idf = tf.gather(idf, q_input)
        grid_p_idf = tf.tile(grid_p_idf, [1, max_a_len])
        grid_p_idf = tf.reshape(grid_p_idf, [batch_size, max_a_len, max_q_len])
        grid_p_idf = tf.expand_dims(grid_p_idf, axis=3, name='expanded_grid2')

        grid_a_idf = tf.gather(idf, a_input)
        grid_a_idf = tf.tile(grid_a_idf, [1, max_q_len])
        grid_a_idf = tf.transpose(tf.reshape(grid_a_idf, [batch_size,  max_q_len,  max_a_len]), perm=[0,2,1])
        grid_a_idf = tf.expand_dims(grid_a_idf, axis=3, name='expanded_grid3')
        """
        grid = tf.concat([gridCos, gridCos2, gridCos3], axis=3)
        #grid = tf.concat([gridCos, grid_p_idf, grid_a_idf], axis=3)
        #grid = gridCos
        # Use tf.image.crop_and_resize() ---> beautiful function
        # Normalise lengths
        a_seqlens = tf.to_float(a_seqlens) / tf.to_float(max_a_len)
        q_seqlens = tf.to_float(q_seqlens) / tf.to_float(max_q_len)
        a_seqlens = tf.expand_dims(a_seqlens, axis=0, name='exp_aseq')
        q_seqlens = tf.expand_dims(q_seqlens, axis=0, name='exp_qseq')
        boxes = tf.concat([a_seqlens, q_seqlens], axis=0)
        zeros = tf.zeros([2, batch_size])
        boxes = tf.concat([zeros, boxes], axis=0)
        boxes = tf.transpose(boxes)
        box_ind = tf.range(batch_size)
        img = tf.image.crop_and_resize(grid, boxes, box_ind, [180, 180])

        # Pass image through Inception network
        with slim.arg_scope(resnet_v2.resnet_arg_scope()) as scope:
              logits, _ = resnet_v2.resnet_v2_152(img, self.network_architecture['n_out'], is_training=is_training)

        logits = tf.squeeze(logits, [2,3])
        probabilities = self.network_architecture['output_fn'](logits)
        predictions = tf.cast(tf.round(probabilities), dtype=tf.float32)


        return predictions, probabilities, logits

    def fit(self,
            train_data,
            valid_data,
            load_path,
            topics,
            topic_lens,
            aug_topics,
            aug_topic_lens,
            aug_topics2,
            aug_topic_lens2,
            aug_topics3,
            aug_topic_lens3,
            aug_topics4,
            aug_topic_lens4,
            aug_topics5,
            aug_topic_lens5,
            aug_topics6,
            aug_topic_lens6,
            aug_topics7,
            aug_topic_lens7,
            aug_topics8,
            aug_topic_lens8,
            aug_topics9,
            aug_topic_lens9,
            sorted_resps,
            sorted_resp_lens,
            prompt_resp_ids,
            prompt_resp_id_lens,
            bert_dists,
            bert_weights,
            arr_unigrams,
            unigram_path,
            train_size=100,
            valid_size=100,
            learning_rate=1e-2,
            lr_decay=0.8,
            dropout=1.0,
            batch_size=50,
            distortion=1.0,
            optimizer=tf.train.AdamOptimizer,
            optimizer_params={},
            n_epochs=30,
            n_samples=1,  # Number of negative samples to generate per positive sample
            epoch=1):
        with self._graph.as_default():
            # Compute number of training examples and batch size
            n_examples = train_size * (1 + n_samples)
            n_batches = n_examples / (batch_size * (1 + n_samples))

            # If some variables have been initialized - get them into a set
            temp = set(tf.global_variables())

            # Define Global step for training
            global_step = tf.Variable(0, trainable=False, name='global_step')

            # Set up inputs
            with tf.variable_scope(self._input_scope, reuse=True) as scope:
                # Construct training data queues
                targets, \
                q_ids, \
                responses, \
                response_lengths, _, _ = self._construct_dataset_from_tfrecord([train_data],
                                                                               self._parse_func,
                                                                               self._map_func,
                                                                               self._batch_func,
                                                                               batch_size,
                                                                               train=True,
                                                                               capacity_mul=1000,
                                                                               num_threads=8)

                valid_iterator = self._construct_dataset_from_tfrecord([valid_data],
                                                                       self._parse_func,
                                                                       self._map_func,
                                                                       self._batch_func,
                                                                       batch_size,
                                                                       train=False,
                                                                       capacity_mul=100,
                                                                       num_threads=1)
                valid_targets, \
                valid_q_ids, \
                valid_responses, \
                valid_response_lengths, _, _ = valid_iterator.get_next(name='valid_data')
        
                # Augment by a factor of 2
                """
                aug_q_ids = self._sample_augment(q_ids=q_ids)
                aug_valid_q_ids = self._sample_augment(q_ids=valid_q_ids)
                
                aug2_q_ids = self._sample_augment(q_ids=q_ids)
                aug2_valid_q_ids = self._sample_augment(q_ids=valid_q_ids)
                aug3_q_ids = self._sample_augment(q_ids=q_ids)
                aug3_valid_q_ids = self._sample_augment(q_ids=valid_q_ids)
                aug4_q_ids = self._sample_augment(q_ids=q_ids)
                aug4_valid_q_ids = self._sample_augment(q_ids=valid_q_ids)
                aug5_q_ids = self._sample_augment(q_ids=q_ids)
                aug5_valid_q_ids = self._sample_augment(q_ids=valid_q_ids)
                aug6_q_ids = self._sample_augment(q_ids=q_ids)
                aug6_valid_q_ids = self._sample_augment(q_ids=valid_q_ids)
                aug7_q_ids = self._sample_augment(q_ids=q_ids)
                aug7_valid_q_ids = self._sample_augment(q_ids=valid_q_ids)
                aug8_q_ids = self._sample_augment(q_ids=q_ids)
                aug8_valid_q_ids = self._sample_augment(q_ids=valid_q_ids)
                aug9_q_ids = self._sample_augment(q_ids=q_ids)
                aug9_valid_q_ids = self._sample_augment(q_ids=valid_q_ids)

                """
                """
                aug_targets, aug_q_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug_q_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug_valid_targets, aug_valid_q_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug_valid_q_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                
                
                aug2_targets, aug2_q_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug2_q_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug2_valid_targets, aug2_valid_q_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug2_valid_q_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug3_targets, aug3_q_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug3_q_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug3_valid_targets, aug3_valid_q_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug3_valid_q_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug4_targets, aug4_q_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug4_q_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug4_valid_targets, aug4_valid_q_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug4_valid_q_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug5_targets, aug5_q_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug5_q_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug5_valid_targets, aug5_valid_q_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug5_valid_q_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug6_targets, aug6_q_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug6_q_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug6_valid_targets, aug6_valid_q_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug6_valid_q_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug7_targets, aug7_q_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug7_q_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug7_valid_targets, aug7_valid_q_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug7_valid_q_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug8_targets, aug8_q_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug8_q_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug8_valid_targets, aug8_valid_q_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug8_valid_q_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug9_targets, aug9_q_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug9_q_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug9_valid_targets, aug9_valid_q_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug9_valid_q_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                """

                targets, q_ids = self._sample_refined(targets=targets,
                                                      q_ids=q_ids,
                                                      batch_size=batch_size,
                                                      n_samples=n_samples,
                                                      arr_unigrams=arr_unigrams,
                                                      p_id_weights=bert_weights)            

                valid_targets, valid_q_ids = self._sample_refined(targets=valid_targets,
                                                                  q_ids=valid_q_ids,
                                                                  batch_size=batch_size,
                                                                  n_samples=n_samples,
                                                                  arr_unigrams=arr_unigrams,
                                                                  p_id_weights=bert_weights)


                # Duplicate list of tensors for negative example generation and data augmentation               
                response_lengths = tf.tile(response_lengths, [n_samples + 1])
                responses = tf.tile(responses, [1 + n_samples, 1])
                valid_response_lengths = tf.tile(valid_response_lengths, [n_samples + 1])
                valid_responses = tf.tile(valid_responses, [1 + n_samples, 1])



            topics = tf.convert_to_tensor(topics, dtype=tf.int32)
            topic_lens = tf.convert_to_tensor(topic_lens, dtype=tf.int32)
            """
            aug_topics = tf.convert_to_tensor(aug_topics, dtype=tf.int32)
            aug_topic_lens = tf.convert_to_tensor(aug_topic_lens, dtype=tf.int32)
            
            aug_topics2 = tf.convert_to_tensor(aug_topics2, dtype=tf.int32)
            aug_topic_lens2 = tf.convert_to_tensor(aug_topic_lens2, dtype=tf.int32)
            aug_topics3 = tf.convert_to_tensor(aug_topics3, dtype=tf.int32)
            aug_topic_lens3 = tf.convert_to_tensor(aug_topic_lens3, dtype=tf.int32)
            aug_topics4 = tf.convert_to_tensor(aug_topics4, dtype=tf.int32)
            aug_topic_lens4 = tf.convert_to_tensor(aug_topic_lens4, dtype=tf.int32)
            aug_topics5 = tf.convert_to_tensor(aug_topics5, dtype=tf.int32)
            aug_topic_lens5 = tf.convert_to_tensor(aug_topic_lens5, dtype=tf.int32)
            aug_topics6 = tf.convert_to_tensor(aug_topics6, dtype=tf.int32)
            aug_topic_lens6 = tf.convert_to_tensor(aug_topic_lens6, dtype=tf.int32)
            aug_topics7 = tf.convert_to_tensor(aug_topics7, dtype=tf.int32)
            aug_topic_lens7 = tf.convert_to_tensor(aug_topic_lens7, dtype=tf.int32)
            aug_topics8 = tf.convert_to_tensor(aug_topics8, dtype=tf.int32)
            aug_topic_lens8 = tf.convert_to_tensor(aug_topic_lens8, dtype=tf.int32)
            aug_topics9 = tf.convert_to_tensor(aug_topics9, dtype=tf.int32)
            aug_topic_lens9 = tf.convert_to_tensor(aug_topic_lens9, dtype=tf.int32)
            """


            prompts = tf.nn.embedding_lookup(topics, q_ids, name='train_prompt_loopkup')
            prompt_lens = tf.gather(topic_lens, q_ids)

            valid_prompts = tf.nn.embedding_lookup(topics, valid_q_ids, name='valid_prompt_loopkup')
            valid_prompt_lens = tf.gather(topic_lens, valid_q_ids)
            """
            aug_prompts = tf.nn.embedding_lookup(aug_topics, aug_q_ids, name='train_prompt_loopkup')
            aug_prompt_lens = tf.gather(aug_topic_lens, aug_q_ids)

            aug_valid_prompts = tf.nn.embedding_lookup(aug_topics, aug_valid_q_ids, name='valid_prompt_loopkup')
            aug_valid_prompt_lens = tf.gather(aug_topic_lens, aug_valid_q_ids)
             
            aug2_prompts = tf.nn.embedding_lookup(aug_topics2, aug2_q_ids, name='train_prompt_loopkup')
            aug2_prompt_lens = tf.gather(aug_topic_lens2, aug2_q_ids)

            aug2_valid_prompts = tf.nn.embedding_lookup(aug_topics2, aug2_valid_q_ids, name='valid_prompt_loopkup')
            aug2_valid_prompt_lens = tf.gather(aug_topic_lens2, aug2_valid_q_ids)

            aug3_prompts = tf.nn.embedding_lookup(aug_topics3, aug3_q_ids, name='train_prompt_loopkup')
            aug3_prompt_lens = tf.gather(aug_topic_lens3, aug3_q_ids)

            aug3_valid_prompts = tf.nn.embedding_lookup(aug_topics3, aug3_valid_q_ids, name='valid_prompt_loopkup')
            aug3_valid_prompt_lens = tf.gather(aug_topic_lens3, aug3_valid_q_ids)

            aug4_prompts = tf.nn.embedding_lookup(aug_topics4, aug4_q_ids, name='train_prompt_loopkup')
            aug4_prompt_lens = tf.gather(aug_topic_lens4, aug4_q_ids)

            aug4_valid_prompts = tf.nn.embedding_lookup(aug_topics4, aug4_valid_q_ids, name='valid_prompt_loopkup')
            aug4_valid_prompt_lens = tf.gather(aug_topic_lens4, aug4_valid_q_ids)

            aug5_prompts = tf.nn.embedding_lookup(aug_topics5, aug5_q_ids, name='train_prompt_loopkup')
            aug5_prompt_lens = tf.gather(aug_topic_lens5, aug5_q_ids)

            aug5_valid_prompts = tf.nn.embedding_lookup(aug_topics5, aug5_valid_q_ids, name='valid_prompt_loopkup')
            aug5_valid_prompt_lens = tf.gather(aug_topic_lens5, aug5_valid_q_ids)

            aug6_prompts = tf.nn.embedding_lookup(aug_topics6, aug6_q_ids, name='train_prompt_loopkup')
            aug6_prompt_lens = tf.gather(aug_topic_lens6, aug6_q_ids)

            aug6_valid_prompts = tf.nn.embedding_lookup(aug_topics6, aug6_valid_q_ids, name='valid_prompt_loopkup')
            aug6_valid_prompt_lens = tf.gather(aug_topic_lens6, aug6_valid_q_ids)

            aug7_prompts = tf.nn.embedding_lookup(aug_topics7, aug7_q_ids, name='train_prompt_loopkup')
            aug7_prompt_lens = tf.gather(aug_topic_lens7, aug7_q_ids)

            aug7_valid_prompts = tf.nn.embedding_lookup(aug_topics7, aug7_valid_q_ids, name='valid_prompt_loopkup')
            aug7_valid_prompt_lens = tf.gather(aug_topic_lens7, aug7_valid_q_ids)

            aug8_prompts = tf.nn.embedding_lookup(aug_topics8, aug8_q_ids, name='train_prompt_loopkup')
            aug8_prompt_lens = tf.gather(aug_topic_lens8, aug8_q_ids)

            aug8_valid_prompts = tf.nn.embedding_lookup(aug_topics8, aug8_valid_q_ids, name='valid_prompt_loopkup')
            aug8_valid_prompt_lens = tf.gather(aug_topic_lens8, aug8_valid_q_ids)

            aug9_prompts = tf.nn.embedding_lookup(aug_topics9, aug9_q_ids, name='train_prompt_loopkup')
            aug9_prompt_lens = tf.gather(aug_topic_lens9, aug9_q_ids)

            aug9_valid_prompts = tf.nn.embedding_lookup(aug_topics9, aug9_valid_q_ids, name='valid_prompt_loopkup')
            aug9_valid_prompt_lens = tf.gather(aug_topic_lens9, aug9_valid_q_ids)
            """

            # Make all prompts tensors of same dimensions
            """
            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros], axis=1))
            aug_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug_prompts, zeros], axis=1), lambda: aug_prompts)
            prompts = tf.concat([prompts, aug_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros], axis=1))
            aug_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug_valid_prompts, zeros], axis=1), lambda: aug_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug_valid_prompt_lens], axis=0)
            
            
            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug2_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros2], axis=1))
            aug2_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug2_prompts, zeros], axis=1), lambda: aug2_prompts)
            prompts = tf.concat([prompts, aug2_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug2_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug2_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(valid_prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros2], axis=1))
            aug2_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug2_valid_prompts, zeros], axis=1), lambda: aug2_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug2_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug2_valid_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug3_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros2], axis=1))
            aug3_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug3_prompts, zeros], axis=1), lambda: aug3_prompts)
            prompts = tf.concat([prompts, aug3_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug3_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug3_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(valid_prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros2], axis=1))
            aug3_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug3_valid_prompts, zeros], axis=1), lambda: aug3_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug3_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug3_valid_prompt_lens], axis=0)


            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug4_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros2], axis=1))
            aug4_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug4_prompts, zeros], axis=1), lambda: aug4_prompts)
            prompts = tf.concat([prompts, aug4_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug4_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug4_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(valid_prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros2], axis=1))
            aug4_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug4_valid_prompts, zeros], axis=1), lambda: aug4_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug4_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug4_valid_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug5_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros2], axis=1))
            aug5_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug5_prompts, zeros], axis=1), lambda: aug5_prompts)
            prompts = tf.concat([prompts, aug5_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug5_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug5_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(valid_prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros2], axis=1))
            aug5_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug5_valid_prompts, zeros], axis=1), lambda: aug5_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug5_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug5_valid_prompt_lens], axis=0)


            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug6_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros2], axis=1))
            aug6_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug6_prompts, zeros], axis=1), lambda: aug6_prompts)
            prompts = tf.concat([prompts, aug6_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug6_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug6_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(valid_prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros2], axis=1))
            aug6_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug6_valid_prompts, zeros], axis=1), lambda: aug6_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug6_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug6_valid_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug7_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros2], axis=1))
            aug7_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug7_prompts, zeros], axis=1), lambda: aug7_prompts)
            prompts = tf.concat([prompts, aug7_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug7_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug7_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(valid_prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros2], axis=1))
            aug7_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug7_valid_prompts, zeros], axis=1), lambda: aug7_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug7_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug7_valid_prompt_lens], axis=0)


            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug8_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros2], axis=1))
            aug8_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug8_prompts, zeros], axis=1), lambda: aug8_prompts)
            prompts = tf.concat([prompts, aug8_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug8_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug8_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(valid_prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros2], axis=1))
            aug8_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug8_valid_prompts, zeros], axis=1), lambda: aug8_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug8_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug8_valid_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug9_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros2], axis=1))
            aug9_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug9_prompts, zeros], axis=1), lambda: aug9_prompts)
            prompts = tf.concat([prompts, aug9_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug9_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug9_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(valid_prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros2], axis=1))
            aug9_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug9_valid_prompts, zeros], axis=1), lambda: aug9_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug9_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug9_valid_prompt_lens], axis=0)


            targets = tf.concat([targets, aug_targets, aug2_targets, aug3_targets, aug4_targets, aug5_targets, aug6_targets, aug7_targets, aug8_targets, aug9_targets], axis=0)
            valid_targets = tf.concat([valid_targets, aug_valid_targets, aug2_valid_targets, aug3_valid_targets, aug4_valid_targets, aug5_valid_targets, aug6_valid_targets, aug7_valid_targets, aug8_valid_targets, aug9_valid_targets], axis=0)


            # Batch size for positive examples has doubled
            batch_size *= 10
            """
            # TEMP augment by factor of 2
            """
            targets = tf.concat([targets, aug_targets], axis=0)
            valid_targets = tf.concat([valid_targets, aug_valid_targets], axis=0)
            batch_size *= 2
            """            

            # Construct Training & Validation models
            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                trn_predictions, \
                trn_probabilities, \
                trn_logits = self._construct_network(a_input=responses,
                                                         a_seqlens=response_lengths,
                                                         q_input=prompts,
                                                         q_seqlens=prompt_lens,
                                                         max_q_len=tf.shape(prompts)[1],
                                                         max_a_len=tf.reduce_max(response_lengths),
                                                         batch_size=batch_size,
                                                         is_training=True,
                                                         keep_prob=self.dropout)

                valid_predictions, \
                valid_probabilities, \
                valid_logits = self._construct_network(a_input=valid_responses,
                                                          a_seqlens=valid_response_lengths,
                                                          q_input=valid_prompts,
                                                          q_seqlens=valid_prompt_lens,
                                                          max_q_len=tf.shape(valid_prompts)[1],
                                                          max_a_len=tf.reduce_max(valid_response_lengths),
                                                          batch_size=batch_size*2,
                                                          keep_prob=1.0)


            # Construct XEntropy training costs
            trn_cost, total_loss = self._construct_xent_cost(targets=targets,
                                                             logits=trn_logits,
                                                             pos_weight=float(n_samples),
                                                             is_training=True)
            evl_cost = self._construct_xent_cost(targets=valid_targets,
                                                 logits=valid_logits,
                                                 pos_weight=float(n_samples),
                                                 is_training=False)

            train_op = util.create_train_op(total_loss=total_loss,
                                            learning_rate=learning_rate,
                                            optimizer=optimizer,
                                            optimizer_params=optimizer_params,
                                            n_examples=n_examples,
                                            batch_size=batch_size,
                                            learning_rate_decay=lr_decay,
                                            global_step=global_step,
                                            clip_gradient_norm=10.0,
                                            summarize_gradients=False)

            # Intialize only newly created variables, as opposed to reused - allows for finetuning and transfer learning :)
            init = tf.variables_initializer(set(tf.global_variables()) - temp)
            self.sess.run(init)
            #self.sess.run(valid_iterator.initializer)

            if load_path != None:
                self._load_variables(load_scope='model/Embeddings/word_embedding',
                                     new_scope='atm/Embeddings/word_embedding', load_path=load_path)

            # Update Log with training details
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = (
                    'Learning Rate: %f\nLearning Rate Decay: %f\nBatch Size: %d\nValid Size: %d\nOptimizer: %s\nDropout: %f\nSEED: %i\n')
                f.write(format_str % (
                    learning_rate, lr_decay, batch_size, valid_size, str(optimizer), dropout, self._seed) + '\n\n')

            format_str = (
                'Epoch %d, Train Loss = %.2f, Valid Loss = %.2f, Valid ROC = %.2f, (%.1f examples/sec; %.3f ' 'sec/batch)')
            print("Starting Training!\n-----------------------------")
            start_time = time.time()
            for epoch in xrange(epoch + 1, epoch + n_epochs + 1):
                # Run Training Loop
                loss = 0.0
                batch_time = time.time()
                print('Total mini-batches')
                print(n_batches)
                for batch in xrange(n_batches):
                    _, loss_value, bubba = self.sess.run([train_op, trn_cost, responses], feed_dict={self.dropout: dropout})
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    loss += loss_value
                    print(loss_value)

                duration = time.time() - batch_time
                loss /= n_batches
                examples_per_sec = batch_size / duration
                sec_per_epoch = float(duration)
                print('YES')

                # Run Validation Loop
                eval_loss = 0.0
                valid_probs = None
                vld_targets = None
                total_size = 0
                self.sess.run(valid_iterator.initializer)
                while True:
                    try:
                        batch_eval_loss, \
                        batch_valid_preds, \
                        batch_valid_probs, \
                        batch_valid_targets = self.sess.run([evl_cost,
                                                             valid_predictions,
                                                             valid_probabilities,
                                                             valid_targets])
                    #print('Got here')
                        size = batch_valid_probs.shape[0]
                        eval_loss += float(size) * batch_eval_loss
                        if valid_probs is None:
                            valid_probs = batch_valid_probs
                            vld_targets = batch_valid_targets
                        else:
                            valid_probs = np.concatenate((valid_probs, batch_valid_probs), axis=0)
                            vld_targets = np.concatenate((vld_targets, batch_valid_targets), axis=0)
                        total_size += size
                    except:  # tf.errors.OutOfRangeError:
                        break

                eval_loss = eval_loss / float(total_size)
                roc_score = roc(np.squeeze(vld_targets), np.squeeze(valid_probs))
                # Temp
                #roc_score=0.0

                # Summarize Epoch
                with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                    f.write(format_str % (epoch, loss, eval_loss, roc_score, examples_per_sec, sec_per_epoch) + '\n')
                print(format_str % (epoch, loss, eval_loss, roc_score, examples_per_sec, sec_per_epoch))
                self.save(step=epoch)

            # Finish Training
            duration = time.time() - start_time
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = ('Training took %.3f sec')
                f.write('\n' + format_str % (duration) + '\n')
                f.write('----------------------------------------------------------\n')
            print(format_str % (duration))
            self.save()

    def predict(self, test_pattern, batch_size=1, cache_inputs=False, apply_bucketing=True):
        """
        Run inference on a trained model on a dataset.
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
                test_logits = self._construct_network(a_input=test_responses,
                                                         a_seqlens=test_response_lengths,
                                                         q_input=test_prompts,
                                                         q_seqlens=test_prompt_lens,
                                                         max_q_len=tf.shape(test_prompts)[1],
                                                         max_a_len=tf.reduce_max(test_response_lengths),
                                                         batch_size=batch_size,
                                                         keep_prob=1.0)
            if batch_size > 1:
                loss = self._construct_xent_cost(targets=test_targets, logits=tf.squeeze(test_logits), pos_weight=1.0,
                                                 is_training=False)
            else:
                loss = self._construct_xent_cost(targets=tf.squeeze(test_targets), logits=tf.squeeze(test_logits), pos_weight=1.0,
                                                 is_training=False)

            self.sess.run(test_iterator.initializer)
            if cache_inputs:
                return self._predict_loop_with_caching(loss, test_probabilities, test_targets,
                                                       test_responses, test_response_lengths, test_prompts,
                                                       test_prompt_lens)
            else:
                return self._predict_loop(loss, test_probabilities, test_targets, test_q_ids)

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

    def _predict_loop(self, loss, test_probabilities, test_targets, test_q_ids):
        test_loss = 0.0
        total_size = 0
        count = 0

        # Variables for storing the batch_ordered data
        while True:
            try:
                batch_eval_loss, \
                batch_test_probs, \
                batch_test_targets, batch_ids = self.sess.run([loss,
                                                    test_probabilities,
                                                    test_targets, test_q_ids])

                print(batch_ids)
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


