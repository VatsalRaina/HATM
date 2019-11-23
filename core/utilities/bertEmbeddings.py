#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import bert.modeling as modeling
import bert.tokenization as tokenization
from sklearn.metrics.pairwise import cosine_similarity

class BertEmbeddings(object):
    def __init__(self):
        self.sess = tf.Session()
        self.input_ids = tf.placeholder(tf.int32, [None, None], name="input_ids")
        self.input_mask = tf.placeholder(tf.int32, [None, None], name="input_mask")

    def create_model(self):
        config = modeling.BertConfig.from_json_file('/home/alta/relevance/vr311/uncased_L-12_H-768_A-12/bert_config.json')
        model = modeling.BertModel(config=config, is_training=False, input_ids=self.input_ids, input_mask=self.input_mask)
        pooled_output = model.get_pooled_output()
        self.pooled_output = pooled_output

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        init_checkpoint = "/home/alta/relevance/vr311/uncased_L-12_H-768_A-12/bert_model.ckpt"
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        self.sess.run(tf.global_variables_initializer())

    def get_bert_dist(self, pr1, pr2):
        tokenizer = tokenization.FullTokenizer(vocab_file = '/home/alta/relevance/vr311/uncased_L-12_H-768_A-12/vocab.txt')
        tokens1 = tokenizer.tokenize(pr1)
        tokens2 = tokenizer.tokenize(pr2)
        tok_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
        tok_ids2 = tokenizer.convert_tokens_to_ids(tokens2)
        ids = []
        ids.append(tok_ids1)
        ids.append(tok_ids2)
  
        max_tokens = 0
        for toks in ids:
            if max_tokens < len(toks):
                max_tokens = len(toks)

        # Pad with 0s the shorter lists and create the corresponding mask
        mask = []
        for i, toks in enumerate(ids):
            ones = [1] * len(toks)
            mask.append(ones)
            if len(toks) < max_tokens:
                zeros = [0] * (max_tokens-len(toks))
                ids[i].extend(zeros)
                mask[i].extend(zeros)

        feed_dict = {self.input_ids: ids, self.input_mask: mask}
        ans = self.sess.run(self.pooled_output, feed_dict=feed_dict)

        #dist = np.linalg.norm(ans[1]-ans[0])
        # Get cosine distance between vectors
        a = ans[0].reshape(1,768)
        b = ans[1].reshape(1,768)
        dist = cosine_similarity(a,b)[0][0]
        return dist
