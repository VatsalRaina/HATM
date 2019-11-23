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
                self.attention, _, _, _, _, _, _, _, _, _ = self._construct_network(a_input=self.x_a,
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


    def _construct_network(self, a_input, a_seqlens, n_samples, p_input, p_seqlens, maxlen, p_ids, batch_size, is_training=False, run_prompt_encoder=False, keep_prob=1.0):
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

        print("p_ids: ")
        print(p_ids.shape)
        print("p_inputs: ")
        print(p_inputs.shape)


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

        print("prompt_embeddings: ")
        print(prompt_embeddings.shape)
        print("keys: ")
        print(keys.shape)


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

            if is_training:
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
                                                     query=attended_prompt_embedding,
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

        # Only the first 4 are in actual code, the rest are returned TEMPorarily
        return predictions, probabilities, logits, prompt_attention, p_inputs, p_inputs_fw, prompt_embeddings, keys, mems, tkeys, tmp, a, attended_prompt_embedding

    def fit(self,
            train_data,
            valid_data,
            load_path,
            topics,
            topic_lens,
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

                bert_dists = tf.convert_to_tensor(bert_dists, dtype=tf.float32)

                targets_o, p_ids_o = self._sampling_function(targets=targets,
                                                         q_ids=p_ids,
                                                         unigram_path=unigram_path,
                                                         batch_size=batch_size,
                                                         n_samples=n_samples,
                                                         name='train',
                                                         bert_dists=bert_dists,
                                                         distortion=distortion)

                targets, p_ids = self._sample_refined(targets=targets,
                                                      q_ids=p_ids,
                                                      batch_size=batch_size,
                                                      n_samples=n_samples,
                                                      arr_unigrams=arr_unigrams,
                                                      p_id_weights=bert_weights)               
                                         
                valid_targets_o, valid_p_ids_o = self._sampling_function(targets=valid_targets,
                                                                     q_ids=valid_p_ids,
                                                                     unigram_path=unigram_path,
                                                                     batch_size=batch_size,
                                                                     n_samples=n_samples,
                                                                     name='valid',
                                                                     bert_dists=bert_dists,
                                                                     distortion=1.0)
                valid_targets, valid_p_ids = self._sample_refined(targets=valid_targets,
                                                                  q_ids=valid_p_ids,
                                                                  batch_size=batch_size,
                                                                  n_samples=n_samples,
                                                                  arr_unigrams=arr_unigrams,
                                                                  p_id_weights=bert_weights)    


            topics = tf.convert_to_tensor(topics, dtype=tf.int32)
            topic_lens = tf.convert_to_tensor(topic_lens, dtype=tf.int32)

            prompts = tf.nn.embedding_lookup(topics, p_ids, name='train_prompot_loopkup')
            prompt_lens = tf.gather(topic_lens, p_ids)

            valid_prompts = tf.nn.embedding_lookup(topics, valid_p_ids, name='valid_prompot_loopkup')
            valid_prompt_lens = tf.gather(topic_lens, valid_p_ids)

            # Construct Training & Validation models
            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                trn_predictions, \
                trn_probabilities, \
                trn_logits, _, x1, x2, x3, x4, x5, x6, x7, x8, x9 = self._construct_network(a_input=responses,
                                                         a_seqlens=response_lengths,
                                                         n_samples=n_samples,
                                                         p_input=prompts,
                                                         p_seqlens=prompt_lens,
                                                         p_ids=p_ids,
                                                         maxlen=tf.reduce_max(response_lengths),
                                                         batch_size=batch_size,
                                                         is_training=True,
                                                         keep_prob=self.dropout)

                valid_predictions, \
                valid_probabilities, \
                valid_logits, \
                valid_attention, _, _, _, _, _, _, _, _, _ = self._construct_network(a_input=valid_responses,
                                                          a_seqlens=valid_response_lengths,
                                                          n_samples=n_samples,
                                                          p_input=valid_prompts,
                                                          p_ids=p_ids,
                                                          p_seqlens=valid_prompt_lens,
                                                          maxlen=tf.reduce_max(valid_response_lengths),
                                                          is_training=False,
                                                          batch_size=batch_size,
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

            variables1 =tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='.*((PROMPT_ATN)|(RNN_KEY)).*')
            variables2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='.*((PROMPT_ATN)|(RNN_KEY)|(Attention)).*')
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


            # Intialize only newly created variables, as opposed to reused - allows for finetuning and transfer learning :)
            init = tf.variables_initializer(set(tf.global_variables()) - temp)
            self.sess.run(init)

            if load_path != None:
                print 'Loading ATM model parameters'
                self._load_variables(load_scope='atm/Embeddings/word_embedding',
                                     new_scope='atm/Embeddings/word_embedding', load_path=load_path, trainable=False)
                #self._load_variables(load_scope='RNN_Q_FW', new_scope='RNN_Q_FW', load_path=load_path, trainable=True)
                #self._load_variables(load_scope='RNN_Q_BW', new_scope='RNN_Q_BW', load_path=load_path, trainable=True)
                self._load_variables(load_scope='RNN_A_FW', new_scope='RNN_A_FW', load_path=load_path, trainable=True)
                self._load_variables(load_scope='RNN_A_BW', new_scope='RNN_A_BW', load_path=load_path, trainable=True)
                self._load_variables(load_scope='Attention', new_scope='Attention', load_path=load_path,
                                    trainable=True)
                self._load_variables(load_scope='Grader', new_scope='Grader', load_path=load_path, trainable=True)

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
                    if epoch <= 2:
                        _, loss_value, x1e,x2e,x3e,x4e,x5e,x6e,x7e,x8e,x9e, p2 = self.sess.run([train_op_new, trn_cost, x1,x2,x3,x4,x5,x6,x7,x8,x9, prompts], feed_dict={self.dropout: dropout})
                       # print("p_ids size: ")
                       # print(p1.shape)
                       # print("p_ids value: ")
                       # print(p1)
                        print("prompts size: ")
                        print(p2.shape)
                       # print("prompts value ")
                       # print(p2)
                       # print("p_inputs: ")
                       # print(x1e.shape)
                       # print("p_inputs_fw: ")
                       # print(x2e.shape)
                       # print("prompt_embeddings: ")
                       # print(x3e.shape)
                       # print("keys: ")
                       # print(x4e.shape)
                       # print("mems: ")
                       # print(x5e.shape)
                       # print("tkeys: ")
                       # print(x6e.shape)
                       # print("tmp: ")
                       # print(x7e.shape)
                       # print("a: ")
                       # print(x8e.shape)
                       # print("attended_prompt_embedding: ")
                       # print(x9e.shape)
                    elif epoch == 3:
                        _, loss_value = self.sess.run([train_op_atn, trn_cost], feed_dict={self.dropout: dropout})
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
                test_attention, _, _, _, _, _, _, _, _, _ = self._construct_network(a_input=test_responses,
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

                print("Batch test prompts shape")
                print(batch_test_prompts.shape)
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
