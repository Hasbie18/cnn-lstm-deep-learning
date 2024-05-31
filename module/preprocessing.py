import numpy as np
import tensorflow as tf
import keras as keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


texts = [
    "The cat sat on the mat",
    "The dog sat on the mat",
    "The cat ate the mat",
]

labels = [0, 1, 0]

def clean_text(text):
    text = text.lower()
    text = text.replace('.', '')
    return text

texts = [clean_text(text) for text in texts]

#tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

#Padding
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

#spread data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

#vector embedding
embedding_dim = 50
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

#inisialisasi embedding secara acak (pre-trained)
embedding_matrix = np.random.randn(vocab_size, embedding_dim)

#CNN-LSTM Model untuk pre-processing

def create_CNN_LSTM_model(input_shape, vocab_size, embedding_dim, embedding_matrix):
    inputs = keras.Input(shape=input_shape)
    embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=input_shape[0], trainable=True)(inputs)
    conv = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(embedding)
    pool = keras.layers.MaxPooling1D(pool_size=2)(conv)
    lstm = keras.layers.LSTM(64)(pool)
    dense = keras.layers.Dense(1, activation='sigmoid')(lstm)
    model = tf.keras.Model(inputs=inputs, outputs=dense)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

input_shape = (max_sequence_length,)
model = create_CNN_LSTM_model(input_shape, vocab_size, embedding_dim, embedding_matrix)

model.summary()