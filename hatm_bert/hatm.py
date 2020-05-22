import os
import time

import matplotlib
import numpy as np

matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score as roc

import tensorflow as tf
import tensorflow.contrib.slim as slim

from core.basemodel import BaseModel
import core.utilities.utilities as util


class HierarchicialAttentionTopicModel(BaseModel):
    def __init__(self, network_architecture=None, name=None, save_path='./', load_path=None, debug_mode=0, seed=100,
                 epoch=None):

        BaseModel.__init__(self, network_architecture=network_architecture, seed=seed, name=name, save_path=save_path,
                           load_path=load_path, debug_mode=debug_mode)

        with self._graph.as_default():
            with tf.variable_scope('input') as scope:
                self._input_scope = scope
                self.x_a = tf.placeholder(tf.int32, [None, None])
                self.x_p = tf.placeholder(tf.int32, [None, None])
                self.plens = tf.placeholder(tf.int32, [None])
                self.p_ids = tf.placeholder(tf.int32, [None])
                self.p_ids_lens = tf.placeholder(tf.int32, [None])
                self.p_ids_seq = tf.placeholder(tf.int32, [None, None])
                self.alens = tf.placeholder(tf.int32, [None])
                self.y = tf.placeholder(tf.float32, [None, 1])
                self.maxlen = tf.placeholder(dtype=tf.int32, shape=[])

                self.dropout = tf.placeholder(tf.float32, [])
                self.batch_size = tf.placeholder(tf.int32, [])

            with tf.variable_scope('atm') as scope:
                prompt_embeddings=np.loadtxt(os.path.join('./model/prompt_embeddings.txt'),dtype=np.float32)
                self.prompt_embeddings= tf.constant(prompt_embeddings,dtype=tf.float32)
                self._model_scope = scope
                self._predictions, \
                self._probabilities, \
                self._logits, \
                self.attention  = self._construct_network(a_input=self.x_a,
                                                          a_seqlens=self.alens,
                                                          p_input=self.x_p,
                                                          p_seqlens=self.plens,
                                                          n_samples=0,
                                                          maxlen=self.maxlen,
                                                          p_ids=self.p_ids,
                                                          is_training=False,
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

    def _construct_prompt_encoder(self, p_input, p_seqlens, batch_size):
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
                                            trainable=False,
                                            shape=[self.network_architecture['n_in'],
                                                   self.network_architecture['n_ehid']],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            regularizer=slim.l2_regularizer(L2),
                                            device='/GPU:0')

            p_inputs = tf.nn.embedding_lookup(embedding, p_input, name='embedded_data')

            p_inputs_fw = tf.transpose(p_inputs, [1, 0, 2])
            p_inputs_bw = tf.transpose(tf.reverse_sequence(p_inputs, seq_lengths=p_seqlens, seq_axis=1, batch_axis=0),
                                       [1, 0, 2])


        prompt_embeddings = self.prompt_embeddings

        with tf.variable_scope('RNN_KEY_FW', initializer=initializer(self._seed)) as scope:
            rnn_fw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            _, state_fw = rnn_fw(p_inputs_fw, sequence_length=p_seqlens, dtype=tf.float32)
        with tf.variable_scope('RNN_KEY_BW', initializer=initializer(self._seed)) as scope:
            rnn_bw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            _, state_bw = rnn_bw(p_inputs_bw, sequence_length=p_seqlens, dtype=tf.float32)


        keys = tf.concat([state_fw[1], state_bw[1]], axis=1)

        with tf.variable_scope('PROMPT_ATN', initializer=initializer(self._seed)) as scope:
            # Compute Attention over known questions
            mems = slim.fully_connected(prompt_embeddings,
                                        2 * self.network_architecture['n_phid'],
                                        activation_fn=None,
                                        weights_regularizer=slim.l2_regularizer(L2),
                                        scope="mem")
            mems = tf.expand_dims(mems, axis=0, name='expanded_mems')
            tkeys = slim.fully_connected(keys,
                                         2 * self.network_architecture['n_phid'],
                                         activation_fn=None,
                                         weights_regularizer=slim.l2_regularizer(L2),
                                         scope="tkeys")
            tkeys = tf.expand_dims(tkeys, axis=1, name='expanded_mems')
            v = slim.model_variable('v',
                                    shape=[2 * self.network_architecture['n_phid'], 1],
                                    regularizer=slim.l2_regularizer(L2),
                                    device='/GPU:0')

            tmp = tf.nn.tanh(mems + tkeys)
            tmp = tf.reshape(tmp, shape=[-1, 2 *self.network_architecture['n_phid']])
            a = tf.exp(tf.reshape(tf.matmul(tmp, v), [batch_size, -1]))

            prompt_attention = a / tf.reduce_sum(a, axis=1, keep_dims=True)
            attended_prompt_embedding = tf.matmul(prompt_attention, prompt_embeddings)

            return attended_prompt_embedding, prompt_attention


    def _construct_network(self, a_input, a_seqlens, n_samples, p_input, p_seqlens, maxlen, p_ids, batch_size, is_training=False, is_adversarial=False, run_prompt_encoder=False, keep_prob=1.0):
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
                                            trainable=True,
                                            shape=[self.network_architecture['n_in'],
                                                   self.network_architecture['n_ehid']],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            regularizer=slim.l2_regularizer(L2),
                                            device='/GPU:0')


            a_inputs = tf.nn.dropout(tf.nn.embedding_lookup(embedding, a_input, name='embedded_data'),
                                     keep_prob=keep_prob, seed=self._seed + 1)
            p_inputs = tf.nn.dropout(tf.nn.embedding_lookup(embedding, p_input, name='embedded_data'),
                                     keep_prob=keep_prob, seed=self._seed + 2)

            p_inputs_fw = tf.transpose(p_inputs, [1, 0, 2])
            p_inputs_bw = tf.transpose(tf.reverse_sequence(p_inputs, seq_lengths=p_seqlens, seq_axis=1, batch_axis=0),
                                       [1, 0, 2])

            a_inputs_fw = tf.transpose(a_inputs, [1, 0, 2])
            a_inputs_bw = tf.transpose(tf.reverse_sequence(a_inputs, seq_lengths=a_seqlens, seq_axis=1, batch_axis=0),
                                       [1, 0, 2])


        if run_prompt_encoder == True:
            # Prompt Encoder RNN
            with tf.variable_scope('RNN_Q_FW', initializer=initializer(self._seed)) as scope:
                rnn_fw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
                _, state_fw = rnn_fw(p_inputs_fw, sequence_length=p_seqlens, dtype=tf.float32)
            with tf.variable_scope('RNN_Q_BW', initializer=initializer(self._seed)) as scope:
                rnn_bw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
                _, state_bw = rnn_bw(p_inputs_bw, sequence_length=p_seqlens, dtype=tf.float32)

            prompt_embeddings = tf.concat([state_fw[1], state_bw[1]], axis=1)
            prompt_embeddings = tf.nn.dropout(prompt_embeddings, keep_prob=keep_prob, seed=self._seed)

        else:
            prompt_embeddings = tf.nn.dropout(self.prompt_embeddings, keep_prob=keep_prob, seed=self._seed)

        with tf.variable_scope('RNN_KEY_FW', initializer=initializer(self._seed)) as scope:
            rnn_fw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            _, state_fw = rnn_fw(p_inputs_fw, sequence_length=p_seqlens, dtype=tf.float32)
        with tf.variable_scope('RNN_KEY_BW', initializer=initializer(self._seed)) as scope:
            rnn_bw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            _, state_bw = rnn_bw(p_inputs_bw, sequence_length=p_seqlens, dtype=tf.float32)


        keys = tf.nn.dropout(tf.concat([state_fw[1], state_bw[1]], axis=1), keep_prob=keep_prob, seed=self._seed + 10)


        with tf.variable_scope('PROMPT_ATN', initializer=initializer(self._seed)) as scope:
            # Compute Attention over known questions
            mems = slim.fully_connected(prompt_embeddings,
                                        2 * self.network_architecture['n_phid'],
                                        activation_fn=None,
                                        weights_regularizer=slim.l2_regularizer(L2),
                                        scope="mem")
            mems = tf.expand_dims(mems, axis=0, name='expanded_mems')
            tkeys = slim.fully_connected(keys,
                                         2 * self.network_architecture['n_phid'],
                                         activation_fn=None,
                                         weights_regularizer=slim.l2_regularizer(L2),
                                         scope="tkeys")
            tkeys = tf.expand_dims(tkeys, axis=1, name='expanded_mems')
            v = slim.model_variable('v',
                                    shape=[2 * self.network_architecture['n_phid'], 1],
                                    regularizer=slim.l2_regularizer(L2),
                                    device='/GPU:0')

            tmp = tf.nn.tanh(mems + tkeys)
           # print tmp.get_shape()
            tmp = tf.nn.dropout(tf.reshape(tmp, shape=[-1, 2 *self.network_architecture['n_phid']]), keep_prob=keep_prob, seed=self._seed + 3)
            a = tf.exp(tf.reshape(tf.matmul(tmp, v), [batch_size * (n_samples + 1), -1]))

            if (is_training or is_adversarial):
                mask = tf.where(tf.equal(tf.expand_dims(p_ids, axis=1),
                                         tf.tile(tf.expand_dims(tf.range(0, self.network_architecture['n_topics'], dtype=tf.int32), axis=0),
                                                 [batch_size * (n_samples + 1), 1])),
                                tf.zeros(shape=[batch_size * (n_samples + 1), self.network_architecture['n_topics']], dtype=tf.float32),
                                tf.ones(shape=[batch_size * (n_samples + 1), self.network_architecture['n_topics']], dtype=tf.float32))
                a = a * mask
                # Draw the prompt attention keep probability from a uniform distribution
                floor = 0.05
                ceil = 0.95
                attention_keep_prob = tf.random_uniform(shape=(), minval=floor, maxval=ceil, seed=self._seed)
                a = tf.nn.dropout(a, attention_keep_prob, seed=self._seed)
            prompt_attention = a / tf.reduce_sum(a, axis=1, keep_dims=True)
            attended_prompt_embedding = tf.matmul(prompt_attention, prompt_embeddings)

        """
        This is a temporary measure to compute the shortest cosine distance between the test prompt embedding and the set of trained prompt embeddings
        
        # Assume batch size is 1 for this to work
        temp_attended_prompt_embeddings = tf.squeeze(attended_prompt_embeddings)
        temp_attended_prompt_embeddings = tf.tile(temp_attended_prompt_embeddings, [379])
        temp_attended_prompt_embeddings = tf.reshape(temp_attended_prompt_embeddings, [379, 2 * self.network_architecture['n_phid']])
        x_norm = tf.nn.l2_normalize(temp_attended_prompt_embeddings, dim=1)
        y_norm = tf.nn.l2_normalize(prompt_embeddings, dim=1)
        cos = tf.reduce_sum(x_norm * y_norm, axis=1)
        cosx = tf.clip_by_value(cos, -1.0, 1.0)
        max_cosx = tf.reduce_max(cosx)       
        """

        # Bias on robust prompt sentence embeddings ---> Very successful when combined with data augmentation!
        """         
        with tf.variable_scope('ADV', initializer=(self._seed)) as scope:
            # Unannotate this section to perform adversarial training
            # Add adversarial vector to be learnt
            adv_embd = slim.model_variable('adversarial',
                                           shape = [2*self.network_architecture['n_phid']],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1),
                                           regularizer=slim.l2_regularizer(L2),
                                           device='/GPU:0' )

            # adv_embd should be broadcasted automatically
            attended_prompt_embedding = tf.add(attended_prompt_embedding, adv_embd)
        """

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

        """
        # This achieves direct data augmentation using a perturbed version of a given prompt embedding 
        with tf.variable_scope('EXT', initializer=(self._seed)) as scope:

            # Try data augmentation with adversarial training where a separate perturbation is learnt for each unique prompt
            adv_embd = slim.model_variable('adversarial',
                                           shape = [self.network_architecture['n_topics'], 2*self.network_architecture['n_phid']],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1),
                                           regularizer=slim.l2_regularizer(L2),
                                           device='/GPU:0' )
            p_adv = tf.nn.embedding_lookup(adv_embd, p_ids, name='prompt_perturbation_lookup')
            embd_new = tf.add(attended_prompt_embedding, p_adv)
            if is_training:
                # Now double batch size
                attended_prompt_embedding = tf.concat((attended_prompt_embedding, embd_new), axis=0)
                batch_size*=2
                a_seqlens = tf.tile(a_seqlens, [2])
                outputs = tf.tile(outputs, [2, 1, 1])
         """
        """
        with tf.variable_scope('ADVERSARIAL', initializer=(self._seed)) as scope:
            adv_embd = slim.model_variable('adversarial',
                                           trainable = True,
                                           shape = [self.network_architecture['n_topics'], 2*self.network_architecture['n_phid']],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1),
                                           regularizer=slim.l2_regularizer(L2),
                                           device='/GPU:0' )
            # Limit perturbation using infinity or 2-norm
            epsilon_adv = 1.0
            #adv_embd = tf.clip_by_value(adv_embd, clip_value_min=-1*epsilon_adv, clip_value_max=epsilon_adv)
            p_adv = tf.nn.embedding_lookup(adv_embd, p_ids, name='prompt_perturbation_lookup')
            p_adv_n = epsilon_adv * p_adv / tf.reduce_sum(p_adv, axis=1, keep_dims=True)
            embd_new = tf.add(attended_prompt_embedding, p_adv_n)
            orig = attended_prompt_embedding
            if (is_training or is_adversarial):
                # Now double batch size
                attended_prompt_embedding = tf.concat((attended_prompt_embedding, embd_new), axis=0)
                # Interpolate additional augmented values
                aug_factor = 18
                for i in range(aug_factor):
                    lam = tf.random_uniform(shape=(), minval=0, maxval=2, seed=self._seed+i)
                    aug_embd = lam*orig + (1-lam)*embd_new
                    attended_prompt_embedding = tf.concat((attended_prompt_embedding, aug_embd), axis=0)
                batch_size*=(2 + aug_factor)
                a_seqlens = tf.tile(a_seqlens, [2+aug_factor])
                outputs = tf.tile(outputs, [2+aug_factor, 1, 1])
        """

        #a_seqlens = tf.tile(a_seqlens, [n_samples + 1])
        #outputs = tf.tile(outputs, [1 + n_samples, 1, 1])
        """
        # Injecting noise and augmenting
        if is_training:
            orig = attended_prompt_embedding
            aug_factor = 9
            for i in range(aug_factor):
                nois = tf.random_normal(shape=[batch_size*2, 2*self.network_architecture['n_phid']], mean=0.0, stddev=0.1, dtype=tf.float32, seed=self._seed+i)
                aug_embd = tf.add(orig, nois)
                attended_prompt_embedding = tf.concat([attended_prompt_embedding, aug_embd], axis=0)
            batch_size*=(1+aug_factor)
            a_seqlens = tf.tile(a_seqlens, [1+aug_factor])
            outputs = tf.tile(outputs, [1+aug_factor,1,1])

        """

        # Implement multi-head attention
 
        hidden, attention = self._bahdanau_attention(memory=outputs, seq_lens=a_seqlens, maxlen=maxlen,
                                                     query=attended_prompt_embedding,
                                                     size=2 * self.network_architecture['n_rhid'],
                                                     batch_size=batch_size * (n_samples + 1),
                                                     idx=0)
        """
        k_multihead = 5
        for i in range(k_multihead-1):
            curr_hidden, curr_attention = self._bahdanau_attention(memory=outputs, seq_lens=a_seqlens, maxlen=maxlen,
                                                         query=attended_prompt_embedding,
                                                         size=2 * self.network_architecture['n_rhid'],
                                                         batch_size=batch_size * (n_samples + 1),
                                                         idx=1+i)
            hidden = tf.concat([hidden, curr_hidden], axis=1)
            attention = tf.concat([attention, curr_attention], axis=1)
        #hidden = hidden / k_multihead
        #attention = attention / k_multihead

        with tf.variable_scope('Resize') as scope:
            resize = slim.model_variable('resize',
                                        shape=[2*self.network_architecture['n_rhid'],
                                        k_multihead*2*self.network_architecture['n_rhid']],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1),
                                        regularizer=slim.l2_regularizer(L2),
                                        device='/GPU:0')
            hidden = tf.matmul(hidden, tf.transpose(resize))
        """

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

        # Only the first 4 are in actual code, the rest are returned TEMPorarily
        return predictions, probabilities, logits, prompt_attention

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
            aug_topics10,
            aug_topic_lens10,
            aug_topics11,
            aug_topic_lens11,
            aug_topics12,
            aug_topic_lens12,
            aug_topics13,
            aug_topic_lens13,
            aug_topics14,
            aug_topic_lens14,
            aug_topics15,
            aug_topic_lens15,
            aug_topics16,
            aug_topic_lens16,
            aug_topics17,
            aug_topic_lens17,
            aug_topics18,
            aug_topic_lens18,
            aug_topics19,
            aug_topic_lens19,
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
            n_samples=1,
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
                p_ids, \
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
                valid_p_ids, \
                valid_responses, \
                valid_response_lengths, _, _ = valid_iterator.get_next(name='valid_data')
                
                      
                # Data augmentation (positive example generation)
                """
                aug_p_ids = self._sample_augment(q_ids=p_ids)
                aug_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)
                aug2_p_ids = self._sample_augment(q_ids=p_ids)
                aug2_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)
                aug3_p_ids = self._sample_augment(q_ids=p_ids)
                aug3_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)
                aug4_p_ids = self._sample_augment(q_ids=p_ids)
                aug4_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)
                aug5_p_ids = self._sample_augment(q_ids=p_ids)
                aug5_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)
                aug6_p_ids = self._sample_augment(q_ids=p_ids)
                aug6_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)
                aug7_p_ids = self._sample_augment(q_ids=p_ids)
                aug7_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)
                aug8_p_ids = self._sample_augment(q_ids=p_ids)
                aug8_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)
                aug9_p_ids = self._sample_augment(q_ids=p_ids)
                aug9_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)
                aug10_p_ids = self._sample_augment(q_ids=p_ids)
                aug10_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)
                aug11_p_ids = self._sample_augment(q_ids=p_ids)
                aug11_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)
                aug12_p_ids = self._sample_augment(q_ids=p_ids)
                aug12_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)
                aug13_p_ids = self._sample_augment(q_ids=p_ids)
                aug13_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)
                aug14_p_ids = self._sample_augment(q_ids=p_ids)
                aug14_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)
                aug15_p_ids = self._sample_augment(q_ids=p_ids)
                aug15_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)
                aug16_p_ids = self._sample_augment(q_ids=p_ids)
                aug16_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)
                aug17_p_ids = self._sample_augment(q_ids=p_ids)
                aug17_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)
                aug18_p_ids = self._sample_augment(q_ids=p_ids)
                aug18_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)
                aug19_p_ids = self._sample_augment(q_ids=p_ids)
                aug19_valid_p_ids = self._sample_augment(q_ids=valid_p_ids)



                aug2_targets, aug2_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug2_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug2_valid_targets, aug2_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug2_valid_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug3_targets, aug3_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug3_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug3_valid_targets, aug3_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug3_valid_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug4_targets, aug4_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug4_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug4_valid_targets, aug4_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug4_valid_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug5_targets, aug5_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug5_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug5_valid_targets, aug5_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug5_valid_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug6_targets, aug6_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug6_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug6_valid_targets, aug6_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug6_valid_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug7_targets, aug7_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug7_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug7_valid_targets, aug7_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug7_valid_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug8_targets, aug8_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug8_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug8_valid_targets, aug8_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug8_valid_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug9_targets, aug9_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug9_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug9_valid_targets, aug9_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug9_valid_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug10_targets, aug10_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug10_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug10_valid_targets, aug10_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug10_valid_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug11_targets, aug11_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug11_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug11_valid_targets, aug11_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug11_valid_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug12_targets, aug12_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug12_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug12_valid_targets, aug12_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug12_valid_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug13_targets, aug13_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug13_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug13_valid_targets, aug13_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug13_valid_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug14_targets, aug14_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug14_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug14_valid_targets, aug14_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug14_valid_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)


                aug15_targets, aug15_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug15_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug15_valid_targets, aug15_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug15_valid_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug16_targets, aug16_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug16_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug16_valid_targets, aug16_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug16_valid_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug17_targets, aug17_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug17_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug17_valid_targets, aug17_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug17_valid_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug18_targets, aug18_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug18_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug18_valid_targets, aug18_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug18_valid_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)
                aug19_targets, aug19_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug19_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)

                aug19_valid_targets, aug19_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                              q_ids=aug19_valid_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)


                
                aug_targets, aug_p_ids = self._sample_refined(targets=targets,
                                                              q_ids=aug_p_ids,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              arr_unigrams=arr_unigrams,
                                                              p_id_weights=bert_weights)               
                                         
                
                aug_valid_targets, aug_valid_p_ids = self._sample_refined(targets=valid_targets,
                                                                          q_ids=aug_valid_p_ids,
                                                                          batch_size=batch_size,
                                                                          n_samples=n_samples,
                                                                          arr_unigrams=arr_unigrams,
                                                                          p_id_weights=bert_weights)
                """
                targets, p_ids = self._sample_refined(targets=targets,
                                                      q_ids=p_ids,
                                                      batch_size=batch_size,
                                                      n_samples=n_samples,
                                                      arr_unigrams=arr_unigrams,
                                                      p_id_weights=bert_weights)               
                                         
                
                valid_targets, valid_p_ids = self._sample_refined(targets=valid_targets,
                                                                  q_ids=valid_p_ids,
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
            aug_topics10 = tf.convert_to_tensor(aug_topics10, dtype=tf.int32)
            aug_topic_lens10 = tf.convert_to_tensor(aug_topic_lens10, dtype=tf.int32)
            aug_topics11 = tf.convert_to_tensor(aug_topics11, dtype=tf.int32)
            aug_topic_lens11 = tf.convert_to_tensor(aug_topic_lens11, dtype=tf.int32)
            aug_topics12 = tf.convert_to_tensor(aug_topics12, dtype=tf.int32)
            aug_topic_lens12 = tf.convert_to_tensor(aug_topic_lens12, dtype=tf.int32)
            aug_topics13 = tf.convert_to_tensor(aug_topics13, dtype=tf.int32)
            aug_topic_lens13 = tf.convert_to_tensor(aug_topic_lens13, dtype=tf.int32)
            aug_topics14 = tf.convert_to_tensor(aug_topics14, dtype=tf.int32)
            aug_topic_lens14 = tf.convert_to_tensor(aug_topic_lens14, dtype=tf.int32)
            aug_topics15 = tf.convert_to_tensor(aug_topics15, dtype=tf.int32)
            aug_topic_lens15 = tf.convert_to_tensor(aug_topic_lens15, dtype=tf.int32)
            aug_topics16 = tf.convert_to_tensor(aug_topics16, dtype=tf.int32)
            aug_topic_lens16 = tf.convert_to_tensor(aug_topic_lens16, dtype=tf.int32)
            aug_topics17 = tf.convert_to_tensor(aug_topics17, dtype=tf.int32)
            aug_topic_lens17 = tf.convert_to_tensor(aug_topic_lens17, dtype=tf.int32)
            aug_topics18 = tf.convert_to_tensor(aug_topics18, dtype=tf.int32)
            aug_topic_lens18 = tf.convert_to_tensor(aug_topic_lens18, dtype=tf.int32)
            aug_topics19 = tf.convert_to_tensor(aug_topics19, dtype=tf.int32)
            aug_topic_lens19 = tf.convert_to_tensor(aug_topic_lens19, dtype=tf.int32)            
            """

            

            prompts = tf.nn.embedding_lookup(topics, p_ids, name='train_prompot_loopkup')
            prompt_lens = tf.gather(topic_lens, p_ids)
            
            valid_prompts = tf.nn.embedding_lookup(topics, valid_p_ids, name='valid_prompot_loopkup')
            valid_prompt_lens = tf.gather(topic_lens, valid_p_ids)
            """
            aug_prompts = tf.nn.embedding_lookup(aug_topics, aug_p_ids, name='train_prompot_loopkup')
            aug_prompt_lens = tf.gather(aug_topic_lens, aug_p_ids)

            aug_valid_prompts = tf.nn.embedding_lookup(aug_topics, aug_valid_p_ids, name='valid_prompot_loopkup')
            aug_valid_prompt_lens = tf.gather(aug_topic_lens, aug_valid_p_ids)

            aug2_prompts = tf.nn.embedding_lookup(aug_topics2, aug2_p_ids, name='train_prompt_loopkup')
            aug2_prompt_lens = tf.gather(aug_topic_lens2, aug2_p_ids)

            aug2_valid_prompts = tf.nn.embedding_lookup(aug_topics2, aug2_valid_p_ids, name='valid_prompt_loopkup')
            aug2_valid_prompt_lens = tf.gather(aug_topic_lens2, aug2_valid_p_ids)

            aug3_prompts = tf.nn.embedding_lookup(aug_topics3, aug3_p_ids, name='train_prompt_loopkup')
            aug3_prompt_lens = tf.gather(aug_topic_lens3, aug3_p_ids)

            aug3_valid_prompts = tf.nn.embedding_lookup(aug_topics3, aug3_valid_p_ids, name='valid_prompt_loopkup')
            aug3_valid_prompt_lens = tf.gather(aug_topic_lens3, aug3_valid_p_ids)

            aug4_prompts = tf.nn.embedding_lookup(aug_topics4, aug4_p_ids, name='train_prompt_loopkup')
            aug4_prompt_lens = tf.gather(aug_topic_lens4, aug4_p_ids)

            aug4_valid_prompts = tf.nn.embedding_lookup(aug_topics4, aug4_valid_p_ids, name='valid_prompt_loopkup')
            aug4_valid_prompt_lens = tf.gather(aug_topic_lens4, aug4_valid_p_ids)

            aug5_prompts = tf.nn.embedding_lookup(aug_topics5, aug5_p_ids, name='train_prompt_loopkup')
            aug5_prompt_lens = tf.gather(aug_topic_lens5, aug5_p_ids)

            aug5_valid_prompts = tf.nn.embedding_lookup(aug_topics5, aug5_valid_p_ids, name='valid_prompt_loopkup')
            aug5_valid_prompt_lens = tf.gather(aug_topic_lens5, aug5_valid_p_ids)

            aug6_prompts = tf.nn.embedding_lookup(aug_topics6, aug6_p_ids, name='train_prompt_loopkup')
            aug6_prompt_lens = tf.gather(aug_topic_lens6, aug6_p_ids)

            aug6_valid_prompts = tf.nn.embedding_lookup(aug_topics6, aug6_valid_p_ids, name='valid_prompt_loopkup')
            aug6_valid_prompt_lens = tf.gather(aug_topic_lens6, aug6_valid_p_ids)

            aug7_prompts = tf.nn.embedding_lookup(aug_topics7, aug7_p_ids, name='train_prompt_loopkup')
            aug7_prompt_lens = tf.gather(aug_topic_lens7, aug7_p_ids)

            aug7_valid_prompts = tf.nn.embedding_lookup(aug_topics7, aug7_valid_p_ids, name='valid_prompt_loopkup')
            aug7_valid_prompt_lens = tf.gather(aug_topic_lens7, aug7_valid_p_ids)

            aug8_prompts = tf.nn.embedding_lookup(aug_topics8, aug8_p_ids, name='train_prompt_loopkup')
            aug8_prompt_lens = tf.gather(aug_topic_lens8, aug8_p_ids)

            aug8_valid_prompts = tf.nn.embedding_lookup(aug_topics8, aug8_valid_p_ids, name='valid_prompt_loopkup')
            aug8_valid_prompt_lens = tf.gather(aug_topic_lens8, aug8_valid_p_ids)

            aug9_prompts = tf.nn.embedding_lookup(aug_topics9, aug9_p_ids, name='train_prompt_loopkup')
            aug9_prompt_lens = tf.gather(aug_topic_lens9, aug9_p_ids)

            aug9_valid_prompts = tf.nn.embedding_lookup(aug_topics9, aug9_valid_p_ids, name='valid_prompt_loopkup')
            aug9_valid_prompt_lens = tf.gather(aug_topic_lens9, aug9_valid_p_ids)
           
            aug10_prompts = tf.nn.embedding_lookup(aug_topics10, aug10_p_ids, name='train_prompt_loopkup')
            aug10_prompt_lens = tf.gather(aug_topic_lens10, aug10_p_ids)

            aug10_valid_prompts = tf.nn.embedding_lookup(aug_topics10, aug10_valid_p_ids, name='valid_prompt_loopkup')
            aug10_valid_prompt_lens = tf.gather(aug_topic_lens10, aug10_valid_p_ids)

            aug11_prompts = tf.nn.embedding_lookup(aug_topics11, aug11_p_ids, name='train_prompt_loopkup')
            aug11_prompt_lens = tf.gather(aug_topic_lens11, aug11_p_ids)

            aug11_valid_prompts = tf.nn.embedding_lookup(aug_topics11, aug11_valid_p_ids, name='valid_prompt_loopkup')
            aug11_valid_prompt_lens = tf.gather(aug_topic_lens11, aug11_valid_p_ids)

            aug12_prompts = tf.nn.embedding_lookup(aug_topics12, aug12_p_ids, name='train_prompt_loopkup')
            aug12_prompt_lens = tf.gather(aug_topic_lens12, aug12_p_ids)

            aug12_valid_prompts = tf.nn.embedding_lookup(aug_topics12, aug12_valid_p_ids, name='valid_prompt_loopkup')
            aug12_valid_prompt_lens = tf.gather(aug_topic_lens12, aug12_valid_p_ids)

            aug13_prompts = tf.nn.embedding_lookup(aug_topics13, aug13_p_ids, name='train_prompt_loopkup')
            aug13_prompt_lens = tf.gather(aug_topic_lens13, aug13_p_ids)

            aug13_valid_prompts = tf.nn.embedding_lookup(aug_topics13, aug13_valid_p_ids, name='valid_prompt_loopkup')
            aug13_valid_prompt_lens = tf.gather(aug_topic_lens13, aug13_valid_p_ids)

            aug14_prompts = tf.nn.embedding_lookup(aug_topics14, aug14_p_ids, name='train_prompt_loopkup')
            aug14_prompt_lens = tf.gather(aug_topic_lens14, aug14_p_ids)

            aug14_valid_prompts = tf.nn.embedding_lookup(aug_topics14, aug14_valid_p_ids, name='valid_prompt_loopkup')
            aug14_valid_prompt_lens = tf.gather(aug_topic_lens14, aug14_valid_p_ids)

            aug15_prompts = tf.nn.embedding_lookup(aug_topics15, aug15_p_ids, name='train_prompt_loopkup')
            aug15_prompt_lens = tf.gather(aug_topic_lens15, aug15_p_ids)

            aug15_valid_prompts = tf.nn.embedding_lookup(aug_topics15, aug15_valid_p_ids, name='valid_prompt_loopkup')
            aug15_valid_prompt_lens = tf.gather(aug_topic_lens15, aug15_valid_p_ids)

            aug16_prompts = tf.nn.embedding_lookup(aug_topics16, aug16_p_ids, name='train_prompt_loopkup')
            aug16_prompt_lens = tf.gather(aug_topic_lens16, aug16_p_ids)

            aug16_valid_prompts = tf.nn.embedding_lookup(aug_topics16, aug16_valid_p_ids, name='valid_prompt_loopkup')
            aug16_valid_prompt_lens = tf.gather(aug_topic_lens16, aug16_valid_p_ids)

            aug17_prompts = tf.nn.embedding_lookup(aug_topics17, aug17_p_ids, name='train_prompt_loopkup')
            aug17_prompt_lens = tf.gather(aug_topic_lens17, aug17_p_ids)

            aug17_valid_prompts = tf.nn.embedding_lookup(aug_topics17, aug17_valid_p_ids, name='valid_prompt_loopkup')
            aug17_valid_prompt_lens = tf.gather(aug_topic_lens17, aug17_valid_p_ids)

            aug18_prompts = tf.nn.embedding_lookup(aug_topics18, aug18_p_ids, name='train_prompt_loopkup')
            aug18_prompt_lens = tf.gather(aug_topic_lens18, aug18_p_ids)

            aug18_valid_prompts = tf.nn.embedding_lookup(aug_topics18, aug18_valid_p_ids, name='valid_prompt_loopkup')
            aug18_valid_prompt_lens = tf.gather(aug_topic_lens18, aug18_valid_p_ids)

            aug19_prompts = tf.nn.embedding_lookup(aug_topics19, aug19_p_ids, name='train_prompt_loopkup')
            aug19_prompt_lens = tf.gather(aug_topic_lens19, aug19_p_ids)

            aug19_valid_prompts = tf.nn.embedding_lookup(aug_topics19, aug19_valid_p_ids, name='valid_prompt_loopkup')
            aug19_valid_prompt_lens = tf.gather(aug_topic_lens19, aug19_valid_p_ids)
            


 
            # Make all prompts tensors of same dimensions
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

            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug10_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros2], axis=1))
            aug10_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug10_prompts, zeros], axis=1), lambda: aug10_prompts)
            prompts = tf.concat([prompts, aug10_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug10_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug10_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(valid_prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros2], axis=1))
            aug10_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug10_valid_prompts, zeros], axis=1), lambda: aug10_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug10_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug10_valid_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug11_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros2], axis=1))
            aug11_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug11_prompts, zeros], axis=1), lambda: aug11_prompts)
            prompts = tf.concat([prompts, aug11_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug11_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug11_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(valid_prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros2], axis=1))
            aug11_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug11_valid_prompts, zeros], axis=1), lambda: aug11_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug11_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug11_valid_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug12_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros2], axis=1))
            aug12_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug12_prompts, zeros], axis=1), lambda: aug12_prompts)
            prompts = tf.concat([prompts, aug12_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug12_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug12_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(valid_prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros2], axis=1))
            aug12_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug12_valid_prompts, zeros], axis=1), lambda: aug12_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug12_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug12_valid_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug13_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros2], axis=1))
            aug13_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug13_prompts, zeros], axis=1), lambda: aug13_prompts)
            prompts = tf.concat([prompts, aug13_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug13_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug13_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(valid_prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros2], axis=1))
            aug13_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug13_valid_prompts, zeros], axis=1), lambda: aug13_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug13_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug13_valid_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug14_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros2], axis=1))
            aug14_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug14_prompts, zeros], axis=1), lambda: aug14_prompts)
            prompts = tf.concat([prompts, aug14_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug14_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug14_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(valid_prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros2], axis=1))
            aug14_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug14_valid_prompts, zeros], axis=1), lambda: aug14_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug14_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug14_valid_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug15_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros2], axis=1))
            aug15_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug15_prompts, zeros], axis=1), lambda: aug15_prompts)
            prompts = tf.concat([prompts, aug15_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug15_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug15_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(valid_prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros2], axis=1))
            aug15_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug15_valid_prompts, zeros], axis=1), lambda: aug15_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug15_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug15_valid_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug16_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros2], axis=1))
            aug16_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug16_prompts, zeros], axis=1), lambda: aug16_prompts)
            prompts = tf.concat([prompts, aug16_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug16_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug16_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(valid_prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros2], axis=1))
            aug16_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug16_valid_prompts, zeros], axis=1), lambda: aug16_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug16_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug16_valid_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug17_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros2], axis=1))
            aug17_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug17_prompts, zeros], axis=1), lambda: aug17_prompts)
            prompts = tf.concat([prompts, aug17_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug17_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug17_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(valid_prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros2], axis=1))
            aug17_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug17_valid_prompts, zeros], axis=1), lambda: aug17_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug17_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug17_valid_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug18_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros2], axis=1))
            aug18_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug18_prompts, zeros], axis=1), lambda: aug18_prompts)
            prompts = tf.concat([prompts, aug18_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug18_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug18_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(valid_prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros2], axis=1))
            aug18_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug18_valid_prompts, zeros], axis=1), lambda: aug18_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug18_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug18_valid_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(prompts)[1], tf.shape(aug19_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            prompts = tf.cond(tf.less(0,num_zeros), lambda: prompts, lambda: tf.concat([prompts, zeros2], axis=1))
            aug19_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug19_prompts, zeros], axis=1), lambda: aug19_prompts)
            prompts = tf.concat([prompts, aug19_prompts], axis=0)
            prompt_lens = tf.concat([prompt_lens, aug19_prompt_lens], axis=0)

            num_zeros = tf.subtract(tf.shape(valid_prompts)[1], tf.shape(aug19_valid_prompts)[1])
            zeros = tf.zeros([batch_size*(n_samples+1), tf.abs(num_zeros)], dtype=tf.int32)
            zeros2 = tf.zeros([tf.shape(valid_prompts)[0], tf.abs(num_zeros)], dtype=tf.int32)
            valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: valid_prompts, lambda: tf.concat([valid_prompts, zeros2], axis=1))
            aug19_valid_prompts = tf.cond(tf.less(0,num_zeros), lambda: tf.concat([aug19_valid_prompts, zeros], axis=1), lambda: aug19_valid_prompts)
            valid_prompts = tf.concat([valid_prompts, aug19_valid_prompts], axis=0)
            valid_prompt_lens = tf.concat([valid_prompt_lens, aug19_valid_prompt_lens], axis=0)






            targets = tf.concat([targets, aug_targets, aug2_targets, aug3_targets, aug4_targets, aug5_targets, aug6_targets, aug7_targets, aug8_targets, aug9_targets, aug10_targets, aug11_targets, aug12_targets, aug13_targets, aug14_targets, aug15_targets, aug16_targets, aug17_targets, aug18_targets, aug19_targets], axis=0)
            valid_targets = tf.concat([valid_targets, aug_valid_targets, aug2_valid_targets, aug3_valid_targets, aug4_valid_targets, aug5_valid_targets, aug6_valid_targets, aug7_valid_targets, aug8_valid_targets, aug9_valid_targets, aug10_valid_targets, aug11_valid_targets, aug12_valid_targets, aug13_valid_targets, aug14_valid_targets, aug15_valid_targets, aug16_valid_targets, aug17_valid_targets, aug18_valid_targets, aug19_valid_targets], axis=0)
             

            # Batch size for positive examples has doubled
            batch_size *= 20

            p_ids = tf.concat([p_ids, aug_p_ids, aug2_p_ids, aug3_p_ids, aug4_p_ids, aug5_p_ids, aug6_p_ids, aug7_p_ids, aug8_p_ids, aug9_p_ids, aug10_p_ids, aug11_p_ids, aug12_p_ids, aug13_p_ids, aug14_p_ids, aug15_p_ids, aug16_p_ids, aug17_p_ids, aug18_p_ids, aug19_p_ids], axis=0)
            valid_p_ids = tf.concat([valid_p_ids, aug_valid_p_ids, aug2_valid_p_ids, aug3_valid_p_ids, aug4_valid_p_ids, aug5_valid_p_ids, aug6_valid_p_ids, aug7_valid_p_ids, aug8_valid_p_ids, aug9_valid_p_ids, aug10_valid_p_ids, aug11_valid_p_ids, aug12_valid_p_ids, aug13_valid_p_ids, aug14_valid_p_ids, aug15_valid_p_ids, aug16_valid_p_ids, aug17_valid_p_ids, aug18_valid_p_ids, aug19_valid_p_ids], axis=0)
            """
            # Construct Training & Validation models
            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                trn_predictions, \
                trn_probabilities, \
                trn_logits, _ = self._construct_network(a_input=responses,
                                                         a_seqlens=response_lengths,
                                                         n_samples=n_samples,
                                                         p_input=prompts,
                                                         p_seqlens=prompt_lens,
                                                         p_ids=p_ids,
                                                         maxlen=tf.reduce_max(response_lengths),
                                                         batch_size=batch_size,
                                                         is_training=True,
                                                         is_adversarial=False,
                                                         keep_prob=self.dropout)

                adv_trn_predictions, \
                adv_trn_probabilities, \
                adv_trn_logits, _ = self._construct_network(a_input=responses,
                                                         a_seqlens=response_lengths,
                                                         n_samples=n_samples,
                                                         p_input=prompts,
                                                         p_seqlens=prompt_lens,
                                                         p_ids=p_ids,
                                                         maxlen=tf.reduce_max(response_lengths),
                                                         batch_size=batch_size,
                                                         is_training=False,
                                                         is_adversarial=True,
                                                         keep_prob=1.0)
                


                valid_predictions, \
                valid_probabilities, \
                valid_logits, \
                valid_attention = self._construct_network(a_input=valid_responses,
                                                          a_seqlens=valid_response_lengths,
                                                          n_samples=n_samples,
                                                          p_input=valid_prompts,
                                                          p_ids=p_ids,
                                                          p_seqlens=valid_prompt_lens,
                                                          maxlen=tf.reduce_max(valid_response_lengths),
                                                          is_training=False,
                                                          is_adversarial=False,
                                                          batch_size=batch_size,
                                                          keep_prob=1.0)

            """ 
            nois_aug = True
            aug_fact = 10
            if nois_aug:
                targets = tf.tile(targets, [aug_fact, 1])
            """
            # Construct XEntropy training costs
            trn_cost, total_loss = self._construct_xent_cost(targets=targets,
                                                             logits=trn_logits,
                                                             pos_weight=float(n_samples),
                                                             is_training=True,
                                                             is_adversarial=False)
            """
            adv_cost, adv_total_loss = self._construct_xent_cost(targets=targets,
                                                         logits=adv_trn_logits,
                                                         pos_weight=float(n_samples),
                                                         is_training=False,
                                                         is_adversarial=True)
            """

            evl_cost = self._construct_xent_cost(targets=valid_targets,
                                                 logits=valid_logits,
                                                 pos_weight=float(n_samples),
                                                 is_training=False,
                                                 is_adversarial=False)
            
            variables1 =tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='.*((PROMPT_ATN)|(RNN_KEY)).*')
            variables2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='.*((PROMPT_ATN)|(RNN_KEY)|(Attention)).*')
            """
            variables1 =tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='.*((PROMPT_ATN)|(RNN_KEY)|(ADV)).*')
            variables2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='.*((PROMPT_ATN)|(RNN_KEY)|(ADV)|(Attention)).*')
             
            variables_adv = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='.*((ADVERSARIAL)).*')
            """ 
            train_op_new = util.create_train_op(total_loss=total_loss,
                                            learning_rate=learning_rate,
                                            optimizer=optimizer,
                                            optimizer_params=optimizer_params,
                                            n_examples=n_examples,
                                            variables_to_train=variables1,
                                            batch_size=batch_size,
                                            learning_rate_decay=lr_decay,
                                            global_step=global_step,
                                            clip_gradient_norm=10.0,
                                            summarize_gradients=False)
            train_op_atn = util.create_train_op(total_loss=total_loss,
                                            learning_rate=learning_rate,
                                            optimizer=optimizer,
                                            optimizer_params=optimizer_params,
                                            n_examples=n_examples,
                                            variables_to_train=variables2,
                                            batch_size=batch_size,
                                            learning_rate_decay=lr_decay,
                                            global_step=global_step,
                                            clip_gradient_norm=10.0,
                                            summarize_gradients=False)
            train_op_all = util.create_train_op(total_loss=total_loss,
                                            learning_rate=learning_rate,
                                            optimizer=optimizer,
                                            optimizer_params=optimizer_params,
                                            n_examples=n_examples,
                                            batch_size=batch_size,
                                            learning_rate_decay=lr_decay,
                                            global_step=global_step,
                                            clip_gradient_norm=10.0,
                                            summarize_gradients=False)
            """ 
            train_op_adv = util.create_train_op(total_loss=adv_total_loss,
                                            learning_rate=learning_rate,
                                            optimizer=optimizer,
                                            optimizer_params=optimizer_params,
                                            n_examples=n_examples,
                                            variables_to_train=variables_adv,
                                            batch_size=batch_size,
                                            learning_rate_decay=lr_decay,
                                            global_step=global_step,
                                            clip_gradient_norm=10.0,
                                            summarize_gradients=False)
            """

            # Intialize only newly created variables, as opposed to reused - allows for finetuning and transfer learning :)
            init = tf.variables_initializer(set(tf.global_variables()) - temp)
            self.sess.run(init)

            if load_path != None:
                print 'Loading ATM model parameters'
                self._load_variables(load_scope='atm/Embeddings/word_embedding',
                                     new_scope='atm/Embeddings/word_embedding', load_path=load_path, trainable=True)
                #self._load_variables(load_scope='RNN_Q_FW', new_scope='RNN_Q_FW', load_path=load_path, trainable=True)
                #self._load_variables(load_scope='RNN_Q_BW', new_scope='RNN_Q_BW', load_path=load_path, trainable=True)
                self._load_variables(load_scope='RNN_A_FW', new_scope='RNN_A_FW', load_path=load_path, trainable=True)
                self._load_variables(load_scope='RNN_A_BW', new_scope='RNN_A_BW', load_path=load_path, trainable=True)
                #k_multihead=5
                #for idx in range(k_multihead):
                    #self._load_variables(load_scope='Attention'+str(idx), new_scope='Attention'+str(idx), load_path=load_path,
                                    #trainable=True)
                self._load_variables(load_scope='Attention', new_scope='Attention', load_path=load_path, trainable=True)
                self._load_variables(load_scope='Grader', new_scope='Grader', load_path=load_path, trainable=True)
                #self._load_variables(load_scope='Resize', new_scope='Resize', load_path=load_path, trainable=True)

            # Update Log with training details
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = (
                    'Learning Rate: %f\nLearning Rate Decay: %f\nBatch Size: %d\nValid Size: %d\nOptimizer: %s\nDropout: %f\nSEED: %i\n')
                f.write(format_str % (
                    learning_rate, lr_decay, batch_size, valid_size, str(optimizer), dropout, self._seed) + '\n\n')

            format_str = (
                'Epoch %d, Train Loss = %.2f, Valid Loss = %.2f, Valid ROC = %.2f, (%.1f examples/sec; %.3f ' 'sec/batch)')
            print "Starting Training!\n-----------------------------"
            start_time = time.time()
            for epoch in xrange(epoch + 1, epoch + n_epochs + 1):
                # Run Training Loop
                print("n_batches: ")
                print(n_batches)
                print("batch_size: ")
                print(batch_size)
                loss = 0.0
                batch_time = time.time()
                for batch in xrange(n_batches):
                    if epoch <= 1:
                        _, loss_value, p2 = self.sess.run([train_op_new, trn_cost, prompts], feed_dict={self.dropout: dropout})
                        #_, adv_loss_value = self.sess.run([train_op_adv, adv_cost], feed_dict={self.dropout: 1.0})
                        print("prompts size: ")
                        print(p2.shape)
                    elif epoch == 2:
                        _, loss_value = self.sess.run([train_op_new, trn_cost], feed_dict={self.dropout: dropout})
                        #_, adv_loss_value = self.sess.run([train_op_adv, adv_cost], feed_dict={self.dropout: 1.0})
                    elif epoch == 3:
                        _, loss_value = self.sess.run([train_op_atn, trn_cost], feed_dict={self.dropout: dropout})
                        #_, adv_loss_value = self.sess.run([train_op_adv, adv_cost], feed_dict={self.dropout: 1.0})
                    elif epoch == 4:
                        _, loss_value = self.sess.run([train_op_atn, trn_cost], feed_dict={self.dropout: dropout})
                        #_, adv_loss_value = self.sess.run([train_op_adv, adv_cost], feed_dict={self.dropout: 1.0})
                    else:
                        _, loss_value = self.sess.run([train_op_all, trn_cost], feed_dict={self.dropout: dropout})
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    loss += loss_value

                duration = time.time() - batch_time
                loss /= n_batches
                examples_per_sec = batch_size / duration
                sec_per_epoch = float(duration)

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
                        batch_attention, \
                        batch_valid_targets = self.sess.run([evl_cost,
                                                             valid_predictions,
                                                             valid_probabilities,
                                                             valid_attention,
                                                             valid_targets])
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

                # Summarize Epoch
                with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                    f.write(format_str % (epoch, loss, eval_loss, roc_score, examples_per_sec, sec_per_epoch) + '\n')
                print (format_str % (epoch, loss, eval_loss, roc_score, examples_per_sec, sec_per_epoch))
                self.save(step=epoch)

            # Finish Training
            duration = time.time() - start_time
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = ('Training took %.3f sec')
                f.write('\n' + format_str % (duration) + '\n')
                f.write('----------------------------------------------------------\n')
            print (format_str % (duration))
            self.save()

    def predict(self, test_pattern, batch_size=20, cache_inputs=False, apply_bucketing=True):
        with self._graph.as_default():
            test_files = tf.gfile.Glob(test_pattern)
            if apply_bucketing:
                batching_function = self._batch_func
            else:
                batching_function = self._batch_func_without_bucket
            test_iterator = self._construct_dataset_from_tfrecord(test_files,
                                                                  self._parse_func,
                                                                  self._map_func,
                                                                  self._batch_func,
                                                                  batch_size=batch_size,
                                                                  train=False,
                                                                  capacity_mul=100,
                                                                  num_threads=1)
            test_targets, \
            test_p_ids, \
            test_responses, \
            test_response_lengths, test_prompts, test_prompt_lens = test_iterator.get_next(name='valid_data')

            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                test_predictions, \
                test_probabilities, \
                test_logits, \
                test_attention = self._construct_network(a_input=test_responses,
                                                         a_seqlens=test_response_lengths,
                                                         n_samples=0,
                                                         p_input=test_prompts,
                                                         p_seqlens=test_prompt_lens,
                                                         p_ids=test_p_ids,
                                                         maxlen=tf.reduce_max(test_response_lengths),
                                                         batch_size=batch_size,
                                                         is_training=False,
                                                         keep_prob=1.0)

            loss = self._construct_xent_cost(targets=test_targets, logits=tf.squeeze(test_logits), pos_weight=1.0,
                                             is_training=False)


            self.sess.run(test_iterator.initializer)
            if cache_inputs:
                return self._predict_loop_with_caching(loss, test_probabilities, test_attention, test_targets,
                                                       test_responses, test_response_lengths, test_prompts,
                                                       test_prompt_lens)
            else:
                return self._predict_loop(loss, test_probabilities,  test_attention, test_targets, test_prompts)

    def _predict_loop_with_caching(self, loss, test_probabilities, test_attention, test_targets, test_responses, test_response_lengths,
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
                batch_test_attention, \
                batch_test_targets, \
                batch_responses, \
                batch_response_lengths, \
                batch_prompts, \
                batch_prompt_lens = self.sess.run([loss,
                                                   test_probabilities,
                                                   test_attention,
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
                    test_attention_arr = batch_test_attention
                    test_response_lens_arr = batch_response_lengths[:, np.newaxis]  # becomes shape: (num_batches, 1)
                    test_prompt_lens_arr = batch_prompt_lens[:, np.newaxis]  # becomes shape: (num_batches, 1)
                else:
                    test_probs_arr = np.concatenate((test_probs_arr, batch_test_probs), axis=0)
                    test_labels_arr = np.concatenate((test_labels_arr, batch_test_targets[:, np.newaxis]), axis=0)
                    test_attention_arr = np.concatenate((test_attention_arr, batch_test_attention), axis=0)
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
                test_attention_arr,
                test_labels_arr.astype(np.int32),
                test_response_lens_arr.astype(np.int32),
                test_prompt_lens_arr.astype(np.int32),
                test_responses_list,
                test_prompts_list)

    def _predict_loop(self, loss, test_probabilities, test_attention, test_targets, test_prompts):
        test_loss = 0.0
        total_size = 0
        count = 0

        # Variables for storing the batch_ordered data
        while True:
            try:
                batch_eval_loss, \
                batch_test_probs, \
                batch_test_attention, \
                batch_test_targets, batch_test_prompts = self.sess.run([loss,
                                                    test_probabilities,
                                                    test_attention,
                                                    test_targets, test_prompts])
                size = batch_test_probs.shape[0]
                test_loss += float(size) * batch_eval_loss
                if count == 0:
                    test_probs_arr = batch_test_probs  # shape: (num_batches, 1)
                    test_attention_arr = batch_test_attention
                    test_labels_arr = batch_test_targets[:, np.newaxis]  # becomes shape: (num_batches, 1)
                else:
                    test_probs_arr = np.concatenate((test_probs_arr, batch_test_probs), axis=0)
                    test_attention_arr = np.concatenate((test_attention_arr, batch_test_attention), axis=0)
                    test_labels_arr = np.concatenate((test_labels_arr, batch_test_targets[:, np.newaxis]), axis=0)

                total_size += size
                count += 1
                print(test_probs_arr)
                print(count)
            except:  # todo: tf.errors.OutOfRangeError:
                break

        test_loss = test_loss / float(total_size)

        return (test_loss,
                test_probs_arr,
                test_attention_arr,
                test_labels_arr.astype(np.int32))

    def get_prompt_embeddings(self, prompts, prompt_lens, save_path):
        with self._graph.as_default():
            batch_size = prompts.shape[0]
            prompts = tf.convert_to_tensor(prompts, dtype=tf.int32)
            prompt_lens = tf.convert_to_tensor(prompt_lens, dtype=tf.int32)

            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                prompt_embeddings, prompt_attention = self._construct_prompt_encoder(p_input=prompts, p_seqlens=prompt_lens, batch_size=batch_size)

            embeddings, attention = self.sess.run([prompt_embeddings, prompt_attention])

            path = os.path.join(save_path, 'prompt_embeddings.txt')
            np.savetxt(path, embeddings)

            path = os.path.join(save_path, 'prompt_attention.txt')
            np.savetxt(path, attention)
