
from utils import score
import numpy as np





class SimpleRNN:
    def __init__(self, c):
        self.vocab_size = c['vocab_size']
        self.embed_size = c['embed_size']
        self.num_units = c['num_units']
        self.num_layers = c['num_layers']
        self.mode = None

    def build_train(self, iterator):
        """
        Args:
            data: data iterator
        """
        self.mode = 'train'
        data = iterator.get_next()
        # self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
        cells = []
        for i in range(self.num_layers):
            cells.append(tf.nn.rnn_cell.BasicLSTMCell(self.num_units))
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        # init_state = self.rnn_cell.zero_state(32, tf.float32)
        # init_state = tf.identity(init_state, 'init_state') #Actually it works without this line. But it can be useful
        # _lstm_state = tf.contrib.rnn.LSTMStateTuple(init_state[0, :, :], init_state[1,:,:])

        self.embed_matrix = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0))
        labels, s1_word, s1_len, s2_word, s2_len = data[0], data[1], data[2], data[3], data[4]
        s1_embed = tf.nn.embedding_lookup(self.embed_matrix, s1_word)
        s2_embed = tf.nn.embedding_lookup(self.embed_matrix, s2_word)
        s1_embed = tf.nn.dropout(s1_embed, 0.2)
        s2_embed = tf.nn.dropout(s2_embed, 0.2)
        self.s1_outputs, self.s1_state = tf.nn.dynamic_rnn(
                cell, s1_embed,
                sequence_length=s1_len, time_major=False, dtype=tf.float32)
        self.s2_outputs, self.s2_state = tf.nn.dynamic_rnn(
                cell, s2_embed,
                sequence_length=s2_len, time_major=False, dtype=tf.float32)
        self.states = tf.concat([self.s1_state[-1].h, self.s2_state[-1].h], axis=1)
        self.W = tf.get_variable("W", [self.num_units*2, 1])
        self.b = tf.get_variable("b", [1])
        self.out = tf.squeeze(tf.matmul(self.states, self.W) + self.b)
        self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=self.out, pos_weight=3.))
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        # clipped_gradients, _ = tf.clip_by_global_norm(
                # gradients, 5.0)
        optimizer = tf.train.AdamOptimizer(0.001)
        self.optimize = optimizer.apply_gradients(
                zip(gradients, params))


    def build_eval(self, iterator):
        """
        Args:
            data: data iterator
        """
        self.mode = 'eval'
        data = iterator.get_next()
        cells = []
        for i in range(self.num_layers):
            cells.append(tf.nn.rnn_cell.BasicLSTMCell(self.num_units))
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        # init_state = self.rnn_cell.zero_state(32, tf.float32)
        # init_state = tf.identity(init_state, 'init_state') #Actually it works without this line. But it can be useful
        # _lstm_state = tf.contrib.rnn.LSTMStateTuple(init_state[0, :, :], init_state[1,:,:])

        self.embed_matrix = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0))
        self.labels, s1_word, s1_len, s2_word, s2_len = data[0], data[1], data[2], data[3], data[4]
        s1_embed = tf.nn.embedding_lookup(self.embed_matrix, s1_word)
        s2_embed = tf.nn.embedding_lookup(self.embed_matrix, s2_word)
        self.s1_outputs, self.s1_state = tf.nn.dynamic_rnn(
                cell, s1_embed,
                sequence_length=s1_len, time_major=False, dtype=tf.float32)

        self.s2_outputs, self.s2_state = tf.nn.dynamic_rnn(
                cell, s2_embed,
                sequence_length=s2_len, time_major=False, dtype=tf.float32)
        self.states = tf.concat([self.s1_state[-1].h, self.s2_state[-1].h], axis=1)
        self.W = tf.get_variable("W", [self.num_units*2, 1])
        self.b = tf.get_variable("b", [1])
        self.out = tf.squeeze(tf.matmul(self.states, self.W) + self.b)
        self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.labels, logits=self.out, pos_weight=3.))
        self.prediction = tf.sigmoid(self.out)

    def train(self, sess):
        assert self.mode == 'train'
        _, loss, o = sess.run([self.optimize, self.loss, self.out])
        return loss

    def eval(self, sess, valid_size):
        assert self.mode == 'eval'
        count = 0
        losses = []
        preds = []
        targets = []
        while count < valid_size:
            try:
                loss, pred, target = sess.run([self.loss, self.prediction, self.labels])
                count += 1
                losses.append(loss)
                preds.append(pred)
                targets.append(target[0])
            except tf.errors.OutOfRangeError:
                break
        loss = np.mean(losses)
        f1, acc, prec, recall = score(preds, targets)
        return {'loss':loss, 'f1':f1, 'acc':acc, 'prec':prec, 'recall':recall}
