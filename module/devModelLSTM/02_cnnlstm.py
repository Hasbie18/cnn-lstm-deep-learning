import rdflib
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from gensim.models import Word2Vec


g = rdflib.Graph()
g.parse("files/maizePestandDisease.rdf")

# Ekstrak triple
triples = []
for subj, pred, obj in g:
    triples.append((subj, pred, obj))

# Display triples
print(triples[:10])

# Create embeddings for entities and predicates
entities = list(set([str(triple[0]) for triple in triples] + [str(triple[2]) for triple in triples]))
predicates = list(set([str(triple[1]) for triple in triples]))

# Combine entities and predicates for training embedding
all_words = entities + predicates

# Train Word2Vec model
model = Word2Vec([all_words], vector_size=100, window=5, min_count=1, workers=4)

# Function to get embedding for entity or predicate
def get_embedding(word):
    return model.wv[word]

# Example of getting embedding for an entity
example_entity = str(triples[0][0])
example_embedding = get_embedding(example_entity)
print(f"Hasil dari embedding \n ------------------------------------ \n {example_embedding} \n ------------------------------------ \n")

# Create tensor data
tensor_data = []
for subj, pred, obj in triples:
    subj_emb = get_embedding(str(subj))
    pred_emb = get_embedding(str(pred))
    obj_emb = get_embedding(str(obj))
    tensor_data.append(np.stack([subj_emb, pred_emb, obj_emb]))

# Convert to numpy array
tensor_data = np.array(tensor_data)