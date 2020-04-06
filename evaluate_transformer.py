import tensorflow as tf
from transformer_utils import Transformer
import tensorflow_datasets as tfds
from keras_utils import create_look_ahead_mask, create_padding_mask
from keras_utils import CustomSchedule

EMBEDDING_DIM = 512
NUM_LAYERS = 6
DFF = 2048
NUM_HEADS = 8
MAX_LENGTH = 20

input_vocab = 'input_tok'
target_vocab = 'target_tok'

input_tok = tfds.features.text.SubwordTextEncoder.load_from_file(input_vocab)
target_tok = tfds.features.text.SubwordTextEncoder.load_from_file(target_vocab)

input_vocab_size = input_tok.vocab_size + 2
target_vocab_size = target_tok.vocab_size + 2
dropout_rate = 0.1

transformer = Transformer(NUM_LAYERS, EMBEDDING_DIM, NUM_HEADS, DFF,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

learning_rate = CustomSchedule(EMBEDDING_DIM)
optimizer = tf.keras.optimizers.Adam(learning_rate,
                                     beta_1=0.9, beta_2=0.98, epsilon=1e-9)

checkpoint_path = './checkpoints'
ckpt = tf.train.Checkpoint(optimizer = optimizer,
                            transformer = transformer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = 5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
else:
    print("No checkpoint found")

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)

    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def evaluate(sentence):
    start_token = [input_tok.vocab_size]
    end_token = [input_tok.vocab_size + 1]

    inp_sentence = start_token + input_tok.encode(sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    decoder_input = [target_tok.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output
        )

        predictions, _ = transformer(encoder_input,
                                    output, 
                                    False, 
                                    enc_padding_mask,
                                    combined_mask,
                                    dec_padding_mask)

        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == target_tok.vocab_size + 1:
            return tf.squeeze(output, axis=0)
        
        output = tf.concat([output, predicted_id], axis=-1)
    
    return tf.squeeze(output, axis=0)

def translate(sentence):
    result = evaluate(sentence)

    predicted_sentence = target_tok.decode([i for i in result if i < target_tok.vocab_size])

    print(f'Input: {sentence}')
    print(f'Predic: {predicted_sentence}')

translate(input('Input:: '))