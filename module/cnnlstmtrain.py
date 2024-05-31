import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return {
            'text': torch.tensor(text, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(CNNLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
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

def main():
    texts = [[1, 2, 3, 4], [2, 3, 4, 5]]  # Example tokenized text
    labels = [0, 1]

    # Create dataset and data loader
    dataset = TextDataset(texts, labels)
    batch_size = 32
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model, optimizer, and criterion
    num_classes = 2
    vocab_size = 10000
    embedding_dim = 128
    hidden_dim = 256
    output_dim = 128
    model = CNNLSTMModel(num_classes, vocab_size, embedding_dim, hidden_dim, output_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train model
    for epoch in range(10):
        loss = train(model, device, train_loader, optimizer, criterion)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    # Test model
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    accuracy = test(model, device, test_loader)
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()
