from enc_dec_utils import Encoder, Decoder
import tensorflow_datasets as tfds
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import time
import io
import re
import os
from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('path', None, 'Path to dataset')
flags.DEFINE_integer('batch', 64, 'Batch Size')
flags.DEFINE_integer('epochs', 100, 'No of Epochs')
flags.DEFINE_integer('patience', 5, 'Patience for early stopping')
flags.DEFINE_integer('sample', 150_000, 'No of lines to train on')
BUFFER_SIZE = 420_000
EMBEDDING_DIM = 256


def preprocess(sentence, lower=False):
    if lower:
        sentence = sentence.lower()
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = sentence.strip()

    sentence = ' '.join(sentence.split())

    return sentence


def create_dataset(path, num_examples, lower=False):
    lines = io.open(path, encoding='utf-8').read().strip().split('\n')
    language1 = []
    language2 = []

    for line in tqdm(lines):
        lang1, lang2, _ = line.split('\t')
        lang1, lang2 = preprocess(lang1, lower), preprocess(lang2, lower)

        language1.append(lang1)
        language2.append(lang2)

    return language1[:num_examples], language2[:num_examples]


def create_tokenizer(lang1, lang2):
    english = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (line for line in lang1), target_vocab_size=2**13,
    )

    french = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (line for line in lang2), target_vocab_size=2**13,
    )

    return english, french


def append_tokens(lang1, lang2, tok1, tok2):
    lang1 = [tok1.vocab_size] + tok1.encode(lang1) + [tok1.vocab_size + 1]
    lang2 = [tok2.vocab_size] + tok2.encode(lang2) + [tok2.vocab_size + 1]

    return lang1, lang2


def load_dataset(path, num_examples):
    lang1, lang2 = create_dataset(path, num_examples=num_examples, lower=True)
    tok1, tok2 = create_tokenizer(lang1, lang2)
    language1, language2 = [], []
    for val1, val2 in tqdm(zip(lang1, lang2)):
        val1, val2 = append_tokens(val1, val2, tok1, tok2)
        language1.append(val1)
        language2.append(val2)

    language1 = tf.keras.preprocessing.sequence.pad_sequences(language1,
                                                              padding='post')
    language2 = tf.keras.preprocessing.sequence.pad_sequences(language2,
                                                              padding='post')

    return language1, language2, tok1, tok2


CRITERION = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none'
)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = CRITERION(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)


def main(absl):
    lang1, lang2, tok1, tok2 = load_dataset(
        FLAGS.path, num_examples=FLAGS.sample)
    dataset = tf.data.Dataset.from_tensor_slices((lang1, lang2))

    dataset = dataset.shuffle(BUFFER_SIZE).batch(FLAGS.batch,
                                                 drop_remainder=True)

    vocab_inp_size = tok1.vocab_size + 2
    vocab_tar_size = tok2.vocab_size + 2
    units = 1024

    encoder = Encoder(vocab_inp_size, EMBEDDING_DIM,
                      units, FLAGS.batch, batch_norm=True)
    decoder = Decoder(vocab_tar_size, EMBEDDING_DIM,
                      units, FLAGS.batch, batch_norm=True)

    optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_path = './checkpoints/train'
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=3)

    loss_history = []
    steps = FLAGS.sample // FLAGS.batch

    @tf.function
    def train_step(inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([tok2.vocab_size] * FLAGS.batch, 1)

            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden,
                                                     enc_output)

                loss += loss_function(targ[:, t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    try:
        print('Training Start...')
        for epoch in range(FLAGS.epochs):
            start = time.time()
            total_loss = 0
            enc_hidden = encoder.initialize_hidden_state()

            print(f'Epoch: {epoch + 1} Started')

            for batch, (inp, targ) in enumerate(dataset):
                batch_loss = train_step(inp, targ, enc_hidden)
                total_loss += batch_loss
                print('.', end='')

            total_loss = (total_loss / steps) / steps
            print(
                f"\nTime: {round(time.time() - start, 2)} Loss: {total_loss}\n")

            if (epoch + 1) % 10 == 0:
                print('Checkpoint Saved')
                ckpt_manager.save()

            loss_history.append(total_loss)
            low = len(np.where(np.array(loss_history) < total_loss)[0])
            if low >= FLAGS.patience:
                print('Early Stopping...')
                break
    except KeyboardInterrupt:
        ckpt_manager.save()
        print('Training End...')
        tok1.save_to_file('tok_lang1')
        tok2.save_to_file('tok_lang2')

    print('Training End...')
    tok1.save_to_file('tok_lang1')
    tok2.save_to_file('tok_lang2')
    ckpt_manager.save()


if __name__ == '__main__':
    app.run(main)
