import numpy as np
from keras.models import Sequential
from keras.layers import Reshape, Conv2D, MaxPooling2D, Flatten, LSTM, Dense
from keras.utils import to_categorical
from rdflib import Graph
from gensim.models import Word2Vec

# Load RDF data
g = Graph()
g.parse("files/maizePestandDisease.rdf")

# Extract entities and predicates from RDF data
entities = set()
predicates = set()
for subj, pred, obj in g:
    entities.add(subj)
    entities.add(obj)
    predicates.add(pred)

# Create entity labels
entity_labels = {entity: i for i, entity in enumerate(entities)}

# Create predicate labels
predicate_labels = {predicate: i for i, predicate in enumerate(predicates)}


# Create Word2Vec model for entity embeddings
entity_sentences = [[str(entity)] for entity in entities]
entity_embeddings = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
entity_embeddings.build_vocab(entity_sentences)
entity_embeddings.train(entity_sentences, total_examples=entity_embeddings.corpus_count, epochs=10)

# Create Word2Vec model for predicate embeddings
predicate_sentences = [[str(predicate)] for predicate in predicates]
predicate_embeddings = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
predicate_embeddings.build_vocab(predicate_sentences)
predicate_embeddings.train(predicate_sentences, total_examples=predicate_embeddings.corpus_count, epochs=10)

# Create tensor data
tensor_data = []
y = []
for subj, pred, obj in g:
    subj_emb = entity_embeddings.wv[str(subj)]
    pred_emb = predicate_embeddings.wv[str(pred)]
    obj_emb = entity_embeddings.wv[str(obj)]
    tensor_data.append([subj_emb, pred_emb, obj_emb])
    y.append(entity_labels[subj])

# Convert tensor data to numpy array
tensor_data = np.array(tensor_data)
tensor_data = tensor_data.reshape((tensor_data.shape[0], 3, 100, 1))

# Convert labels to categorical
y = to_categorical(y)

# Create CNN-LSTM model
model = Sequential()
model.add(Reshape((100, 3, 1), input_shape=(3, 100, 1)))
model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Reshape((-1, 56)))
model.add(LSTM(50))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(tensor_data, y, epochs=50, verbose=1)