import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, batch_size,
                 batch_norm=False):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.units = units
        self.batch_norm = batch_norm

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.norm = tf.keras.layers.BatchNormalization()
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        if self.batch_norm:
            x = self.norm(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.units))


class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_time = tf.expand_dims(query, axis=1)
        score = self.V(tf.nn.tanh(
            self.W1(query_time) + self.W2(values)
        ))

        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, batch_size,
                 batch_norm=False):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.units = units
        self.batch_norm = batch_norm

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.norm = tf.keras.layers.BatchNormalization()
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = Attention(self.units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        if self.batch_norm:
            x = self.norm(x)

        x = tf.concat([tf.expand_dims(context_vector, axis=1), x],
                      axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)

        return x, state, attention_weights
