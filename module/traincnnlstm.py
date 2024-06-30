import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

df = pd.read_csv("files/maize_pest_data.csv")

df['text'] = df['Hama']+''+df['Tanaman']+''+df['Penyakit']+''+df['Gejala']

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Gejala'])

texts = df['text'].values
labels = df['label'].values


#tokenizer atau tokenisasi dalam teks
token = Tokenizer(num_words=10000)
token.fit_on_texts(texts)
sequences = token.texts_to_sequences(texts)
word_index = token.word_index

#padding sequence agar memiliki panjang yang sama
data=pad_sequences(sequences, maxlen=100)

#Memisahkan data menjadi data training dan data uji
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
