import torch
import torch.nn as nn
import gensim.downloader as api
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from rdflib import Graph



class OntologyDataset(Dataset):
    def __init__(self, triplets, labels, vocab, max_len=100):
        self.triplets = triplets
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        text = " ".join(triplet)
        text = text_to_indices(text, self.vocab, self.max_len)
        label = self.labels[idx]
        return {
            'text': torch.tensor(text, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

def tokenize(text):
    return text.lower().split()

def build_vocab(texts, max_vocab_size=10000):
    from collections import Counter
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    most_common = counter.most_common(max_vocab_size)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}
    vocab['<PAD>'] = 0
    return vocab

def text_to_indices(text, vocab, max_len=100):
    tokens = tokenize(text)
    indices = [vocab.get(token, 0) for token in tokens]
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices += [0] * (max_len - len(indices))
    return indices

class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(CNNLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.lstm = nn.LSTM(hidden_dim, output_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in train_loader:
        text = batch['text'].to(device)
        label = batch['label'].to(device)
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            text = batch['text'].to(device)
            label = batch['label'].to(device)
            output = model(text)
            _, predicted = torch.max(output, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)
    return correct / total

def predict(model, device, text, vocab):
    model.eval()
    text = torch.tensor([text_to_indices(text, vocab)]).to(device)
    output = model(text)
    _, predicted = torch.max(output, 1)
    return predicted.item()

def main():
    word_vectors = api.load("glove-wiki-gigaword-100")
    # Load ontology and extract triplets
    g = Graph()
    g.parse("files/maizePestandDisease.rdf")

    triplets = []
    query = """
    SELECT ?subject ?predicate ?object
    WHERE {
      ?subject ?predicate ?object
    }
    """
    results = g.query(query)
    for row in results:
        triplets.append((str(row.subject), str(row.predicate), str(row.object)))

    # Sample labels (replace with actual labels)
    labels = [0, 1, 0, 1] * (len(triplets) // 4)

    # Build vocabulary
    texts = [" ".join(triplet) for triplet in triplets]
    vocab = build_vocab(texts)

    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    train_triplets, test_triplets, train_labels, test_labels = train_test_split(triplets, labels, test_size=0.2)

    # Create datasets and data loaders
    train_dataset = OntologyDataset(train_triplets, train_labels, vocab)
    test_dataset = OntologyDataset(test_triplets, test_labels, vocab)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model, optimizer, and criterion
    num_classes = 2
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, 100))
    embedding_dim = 128
    hidden_dim = 256
    output_dim = 128
    model = CNNLSTMModel(num_classes, vocab_size, embedding_dim, hidden_dim, output_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    for word, idx in vocab.items():
        if word in word_vectors:
            embedding_matrix[idx] = word_vectors[word]
    
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

    # Train model
    for epoch in range(50):
        loss = train(model, device, train_loader, optimizer, criterion)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    # Test model
    accuracy = test(model, device, test_loader)
    print(f'Test Accuracy: {accuracy:.4f}')

    # Predict on new data
    new_text = "Busuk"
    prediction = predict(model, device, new_text, vocab)
    print(f'Prediction: {prediction}')

if __name__ == '__main__':
    main()
