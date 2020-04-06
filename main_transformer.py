from transformer_utils import Transformer
import tensorflow_datasets as tfds
import tensorflow as tf
from tqdm import tqdm
from keras_utils import (CustomSchedule,
                         create_look_ahead_mask,
                         create_padding_mask)
import time
import io
import numpy as np
import re

PATH = './fra.txt'
BUFFER_SIZE = 420_000
BATCH_SIZE = 64
EMBEDDING_DIM = 512
NUM_LAYERS = 6
PATIENCE = 5
DFF = 2048
NUM_HEADS = 8
EXAMPLES = 80_000
EPOCHS = 100


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


def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)

    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def loss_function(real, pred, obj):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = obj(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


lang1, lang2, tok1, tok2 = load_dataset(PATH, num_examples=EXAMPLES)
dataset = tf.data.Dataset.from_tensor_slices((lang1, lang2))

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,
                                             drop_remainder=True)

input_vocab_size = tok1.vocab_size + 2
target_vocab_size = tok2.vocab_size + 2
dropout_rate = 0.1

learning_rate = CustomSchedule(EMBEDDING_DIM)
optimizer = tf.keras.optimizers.Adam(learning_rate,
                                     beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none'
)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

transformer = Transformer(NUM_LAYERS, EMBEDDING_DIM, NUM_HEADS, DFF,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

loss_history = []


@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions, loss_object)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


print("Training Start...")
for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    print(f'Epoch {epoch + 1} Started...')
    for batch, (inp, tar) in enumerate(dataset):
        train_step(inp, tar)
        print(f'Batch: {batch}', end='\r')
    print(f'Time {time.time() - start}')
    print(f'Epoch {epoch + 1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}\n')

    if (epoch + 1) % 10 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving Checkpoint')

    loss_history.append(train_loss.result())
    low = len(np.where(np.array(loss_history) < train_loss.result())[0])
    if low >= PATIENCE:
        print("Early Stopping...")
        break

print("Training End...")
tok1.save_to_file('tok_lang1')
tok2.save_to_file('tok_lang2')
ckpt_manager.save()
