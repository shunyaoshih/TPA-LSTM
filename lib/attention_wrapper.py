import logging
import tensorflow as tf
from tensorflow.layers import dense
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest


class TemporalPatternAttentionMechanism():
    def __call__(self, query, attn_states, attn_size, attn_length,
                 attn_vec_size):
        """
        query: [batch_size, attn_size * 2] (c and h)
        attn_states: [batch_size, attn_length, attn_size] (h)
        new_attns: [batch_size, attn_size]
        new_attn_states: [batch_size, attn_length - 1, attn_size]
        """
        with tf.variable_scope("attention"):
            filter_num = 32
            filter_size = 1

            # w: [batch_size, 1, filter_num]
            w = tf.reshape(
                dense(query, filter_num, use_bias=False), [-1, 1, filter_num])
            reshape_attn_vecs = tf.reshape(attn_states,
                                           [-1, attn_length, attn_size, 1])
            conv_vecs = tf.layers.conv2d(
                inputs=reshape_attn_vecs,
                filters=filter_num,
                kernel_size=[attn_length, filter_size],
                padding="valid",
                activation=None,
            )
            feature_dim = attn_size - filter_size + 1
            # conv_vecs: [batch_size, feature_dim, filter_num]
            conv_vecs = tf.reshape(conv_vecs, [-1, feature_dim, filter_num])

            # s: [batch_size, feature_dim]
            s = tf.reduce_sum(tf.multiply(conv_vecs, w), [2])

            # a: [batch_size, feature_dim]
            a = tf.sigmoid(s)
            # d: [batch_size, filter_num]
            d = tf.reduce_sum(
                tf.multiply(tf.reshape(a, [-1, feature_dim, 1]), conv_vecs),
                [1])
            new_conv_vec = tf.reshape(d, [-1, filter_num])
            new_attns = tf.layers.dense(
                tf.concat([query, new_conv_vec], axis=1), attn_size)
            new_attn_states = tf.slice(attn_states, [0, 1, 0], [-1, -1, -1])
            return new_attns, new_attn_states


class TemporalPatternAttentionCellWrapper(rnn_cell_impl.RNNCell):
    def __init__(self,
                 cell,
                 attn_length,
                 attn_size=None,
                 attn_vec_size=None,
                 input_size=None,
                 state_is_tuple=True,
                 reuse=None):
        """Create a cell with attention.
        Args:
            cell: an RNNCell, an attention is added to it.
            attn_length: integer, the size of an attention window.
            attn_size: integer, the size of an attention vector. Equal to
                cell.output_size by default.
            attn_vec_size: integer, the number of convolutional features
                calculated on attention state and a size of the hidden layer
                built from base cell state. Equal attn_size to by default.
            input_size: integer, the size of a hidden linear layer, built from
                inputs and attention. Derived from the input tensor by default.
            state_is_tuple: If True, accepted and returned states are n-tuples,
                where `n = len(cells)`. By default (False), the states are all
                concatenated along the column axis.
            reuse: (optional) Python boolean describing whether to reuse
                variables in an existing scope. If not `True`, and the existing
                scope already has the given variables, an error is raised.
        Raises:
            TypeError: if cell is not an RNNCell.
            ValueError: if cell returns a state tuple but the flag
                `state_is_tuple` is `False` or if attn_length is zero or less.
        """
        super(TemporalPatternAttentionCellWrapper, self).__init__(_reuse=reuse)
        if not rnn_cell_impl._like_rnncell(cell):
            raise TypeError("The parameter cell is not RNNCell.")
        if nest.is_sequence(cell.state_size) and not state_is_tuple:
            raise ValueError("Cell returns tuple of states, but the flag "
                             "state_is_tuple is not set. State size is: %s" %
                             str(cell.state_size))
        if attn_length <= 0:
            raise ValueError("attn_length should be greater than zero, got %s"
                             % str(attn_length))
        if not state_is_tuple:
            logging.warn(
                "%s: Using a concatenated state is slower and will soon be "
                "deprecated.    Use state_is_tuple=True.", self)
        if attn_size is None:
            attn_size = cell.output_size
        if attn_vec_size is None:
            attn_vec_size = attn_size
        self._state_is_tuple = state_is_tuple
        self._cell = cell
        self._attn_vec_size = attn_vec_size
        self._input_size = input_size
        self._attn_size = attn_size
        self._attn_length = attn_length
        self._reuse = reuse
        self._attention_mech = TemporalPatternAttentionMechanism()

    @property
    def state_size(self):
        size = (self._cell.state_size, self._attn_size,
                self._attn_size * self._attn_length)
        if self._state_is_tuple:
            return size
        else:
            return sum(list(size))

    @property
    def output_size(self):
        return self._attn_size

    def call(self, inputs, state):
        """Long short-term memory cell with attention (LSTMA)."""
        if self._state_is_tuple:
            state, attns, attn_states = state
        else:
            states = state
            state = tf.slice(states, [0, 0], [-1, self._cell.state_size])
            attns = tf.slice(states, [0, self._cell.state_size],
                             [-1, self._attn_size])
            attn_states = tf.slice(
                states, [0, self._cell.state_size + self._attn_size],
                [-1, self._attn_size * self._attn_length])
        attn_states = tf.reshape(attn_states,
                                 [-1, self._attn_length, self._attn_size])
        input_size = self._input_size
        if input_size is None:
            input_size = inputs.get_shape().as_list()[1]
        inputs = dense(
            tf.concat([inputs, attns], 1), input_size, use_bias=True)
        lstm_output, new_state = self._cell(inputs, state)

        if self._state_is_tuple:
            new_state_cat = tf.concat(nest.flatten(new_state), 1)
        else:
            new_state_cat = new_state
        new_attns, new_attn_states = self._attention_mech(
            new_state_cat, attn_states, self._attn_size, self._attn_length,
            self._attn_vec_size)

        with tf.variable_scope("attn_output_projection"):
            output = dense(
                tf.concat([lstm_output, new_attns], 1),
                self._attn_size,
                use_bias=True)

        new_attn_states = tf.concat(
            [new_attn_states, tf.expand_dims(output, 1)], 1)
        new_attn_states = tf.reshape(new_attn_states,
                                     [-1, self._attn_length * self._attn_size])
        new_state = (new_state, new_attns, new_attn_states)
        if not self._state_is_tuple:
            new_state = tf.concat(list(new_state), 1)
        return output, new_state
