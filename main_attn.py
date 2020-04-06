from enc_dec_utils import Encoder, Decoder
import tensorflow_datasets as tfds
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import time
import io
import re
import os

PATH = './fra.txt'
BUFFER_SIZE = 420_000
BATCH_SIZE = 64
EMBEDDING_DIM = 256
EPOCHS = 100
PATIENCE = 5
EXAMPLES = 30_000


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

def loss_function(real, pred, obj):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)


lang1, lang2, tok1, tok2 = load_dataset(PATH, num_examples=EXAMPLES)
dataset = tf.data.Dataset.from_tensor_slices((lang1, lang2))

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,
                                             drop_remainder=True)

vocab_inp_size = tok1.vocab_size + 2
vocab_tar_size = tok2.vocab_size + 2
units = 1024

encoder = Encoder(vocab_inp_size, EMBEDDING_DIM,
                  units, BATCH_SIZE, batch_norm=True)
decoder = Decoder(vocab_tar_size, EMBEDDING_DIM,
                  units, BATCH_SIZE, batch_norm=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none'
)
train_loss = tf.keras.metrics.Mean(name='train_loss')

checkpoint_path = './checkpoints/train'
ckpt = tf.train.Checkpoint(optimizer=optimizer,
                           encoder=encoder,
                           decoder=decoder)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

loss_history = []
steps = len([val for val in dataset]) // BATCH_SIZE


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([tok2.vocab_size] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden,
                                                 enc_output)
            loss += loss_function(targ[:, t], predictions, loss_object)
            dec_input = tf.expand_dims(targ[:, t], 1)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    train_loss(loss / targ.shape[1])


print('Training Start...')
for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()

    print(f'Epoch: {epoch + 1} Started')
    for batch, (inp, targ) in enumerate(dataset):
        train_step(inp, targ, enc_hidden)
        print(f"Batch: {batch}", end='\r')
    print(
        f"\nTime: {round(time.time() - start, 2)} Loss: {train_loss.result()}\n")

    if (epoch + 1) % 10 == 0:
        print('Checkpoint Saved')
        ckpt_save_path = ckpt_manager.save()

    loss_history.append(train_loss.result())
    low = len(np.where(np.array(loss_history) < train_loss.result())[0])
    if low >= PATIENCE:
        print('Early Stopping...')
        break

print('Training End...')
tok1.save_to_file('tok_lang1')
tok2.save_to_file('tok_lang2')
ckpt_manager.save()
