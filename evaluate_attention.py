import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from enc_dec_utils import Encoder, Decoder
import re
from absl import flags, app

FLAGS = flags.FLAGS

def preprocess(sentence, lower=False):
    if lower:
        sentence = sentence.lower()
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = sentence.strip()

    sentence = ' '.join(sentence.split())

    return sentence

max_length = 30
EMBEDDING_DIM = 256
units = 1024

flags.DEFINE_string('input_vocab', None, 'Path to input vocabulary')
flags.DEFINE_string('target_vocab', None, 'Path to target vocabulary')
flags.DEFINE_string('checkpoint', './checkpoints/train', "Path to Checkpoint Directory")

def main(absl):
    input_tok = tfds.features.text.SubwordTextEncoder.load_from_file(FLAGS.input_vocab)
    target_tok = tfds.features.text.SubwordTextEncoder.load_from_file(FLAGS.target_vocab)
    checkpoint_path = FLAGS.checkpoint

    input_vocab_size = input_tok.vocab_size + 2
    target_vocab_size = target_tok.vocab_size + 2

    encoder = Encoder(input_vocab_size, EMBEDDING_DIM,
                units, 1, batch_norm=True)
    decoder = Decoder(target_vocab_size, EMBEDDING_DIM,
                    units, 1, batch_norm=True)

    ckpt = tf.train.Checkpoint(encoder = encoder,
                                decoder = decoder)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
    else:
        print('No Checkpoint Found')

    def evaluate(sentence):
        sentence = preprocess(sentence)
        start_token = [input_tok.vocab_size]
        end_token = [input_tok.vocab_size + 1]

        inputs = tf.convert_to_tensor(start_token + input_tok.encode(sentence) + end_token)
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen = max_length)

        hidden = [tf.zeros((1, units))]
        enc_out, enc_hidden = encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([target_tok.vocab_size], 0)

        result = []

        for t in range(max_length):
            predictions, dec_hidden, _ = decoder(dec_input, 
                                                dec_hidden, 
                                                enc_out)
            
            predicted_id = tf.argmax(predictions[0]).numpy()

            result.append(predicted_id)

            if predicted_id == target_tok.vocab_size + 1:
                return result
            
            dec_input = tf.expand_dims([predicted_id], 0)

        return result

    def translate(sentence):
        result = evaluate(sentence)

        predicted_sentence = target_tok.decode([i for i in result if i < target_tok.vocab_size])

        print(f'Input: {sentence}')
        print(f'Predic: {predicted_sentence}')

    translate(input('Input:: '))


if __name__ == '__main__':
    app.run(main)