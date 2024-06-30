import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Contoh data teks
texts = [
    "Ini adalah contoh teks pertama.",
    "Ini adalah contoh teks kedua.",
    "Teks ketiga merupakan contoh lainnya.",
    "Teks keempat sebagai tambahan.",
    "Ini adalah contoh teks kelima.",
    "Teks keenam merupakan tambahan lainnya."
]


labels = [0, 1, 0, 1, 0, 1]


def clean_text(text):
    text = text.lower()
    text = text.replace('.', '')
    return text

texts = [clean_text(text) for text in texts]

# Tokenisasi
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Membuat dataset TensorFlow
dataset = tf.data.Dataset.from_tensor_slices((padded_sequences, labels))

# Fungsi untuk memisahkan dataset menjadi train dan test
def is_test(x, _):
    return x % 4 == 0

def is_train(x, y):
    return not is_test(x, y)

recover = lambda x, y: y


test_dataset = dataset.enumerate() \
    .filter(is_test) \
    .map(recover)


train_dataset = dataset.enumerate() \
    .filter(is_train) \
    .map(recover)

# Mengatur batching dan prefetching
train_dataset = train_dataset.batch(2).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(2).prefetch(tf.data.experimental.AUTOTUNE)

# Memeriksa data train dan test
for data, label in train_dataset.take(1):
    print("Train data:", data.numpy())
    print("Train label:", label.numpy())

for data, label in test_dataset.take(1):
    print("Test data:", data.numpy())
    print("Test label:", label.numpy())


def create_cnn_lstm_model(input_shape, vocab_size, embedding_dim):
    inputs = tf.keras.Input(shape=input_shape)
    embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_shape[0])(inputs)
    conv = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(embedding)
    pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
    lstm = tf.keras.layers.LSTM(100)(pool)
    dense = tf.keras.layers.Dense(1, activation='sigmoid')(lstm)
    model = tf.keras.Model(inputs=inputs, outputs=dense)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 50
input_shape = (max_sequence_length,)


model = create_cnn_lstm_model(input_shape, vocab_size, embedding_dim)
model.summary()

model.fit(train_dataset, epochs=100, validation_data=test_dataset)