#3维变2维


import tensorflow as tf

class SoftAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        super(SoftAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.w = tf.keras.layers.Dense(self.hidden_dim, activation='tanh')
        self.v = tf.keras.layers.Dense(1)

    def call(self, h):
        w = self.w(h)
        weight = self.v(w)
        weight = tf.squeeze(weight, axis=-1)
        weight = tf.nn.softmax(weight, axis=1)
        weight = tf.expand_dims(weight, axis=-1)
        out = h * weight

        out = tf.reduce_sum(out, axis=1)

        return out