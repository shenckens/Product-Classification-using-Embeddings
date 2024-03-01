import numpy as np
import argparse
import json
import re
import os
import pickle
import torch
import nltk
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt


class LSTM(nn.Module):
    def __init__(self, embedding, num_classes, hidden_dim, num_layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        h_n, c_n = self.lstm(x, (h_0, c_0))
        x = self.fc(h_n[:, -1, :])
        return x


# Push data and model to GPU if available.
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def load_embeddings(path):
    pkl = path[:-3] + "pkl"
    if os.path.exists(pkl):
        print(f"[*] Loading {pkl}")
        with open(pkl, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings
    embeddings = {}
    print(f"[*] Loading {path}")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding = np.array(values[1:], dtype='float32')
            embeddings[word] = embedding
    with open(pkl, "wb") as f:
        print(f"[*] Creating {pkl} for quicker re-accessing.")
        pickle.dump(embeddings, f)
    return embeddings


def load_product_data(path):
    print(f"[*] Loading {path}")
    with open(path, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)
    return data


def sentence_embedding(sentence, embeddings, embedding_dim, max_len):
    if sentence is None:
        return torch.zeros(embedding_dim)
    sentence = sentence.lower()
    regex = r'(?<![0-9])[\.,](?![0-9])|[^\w\s,.]'
    sentence = re.sub(regex, ' ', sentence)
    tokens = nltk.tokenize.word_tokenize(sentence)
    sentence_embedding = [torch.tensor(embeddings[token]) for token in tokens if token in embeddings]
    # Pad the sentence embedding.
    if len(sentence_embedding) < max_len:
        sentence_embedding += [torch.zeros(embedding_dim) for i in range(max_len - len(sentence_embedding))]
    return torch.stack(sentence_embedding)


def test_model(model, dataloader):
    model.eval()
    total_loss = 0
    correct_preds = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.float())
            loss = loss_fn(outputs, labels.float())
            total_loss += loss.item()
            predicted = outputs.sigmoid() > 0.5
            correct_preds += (predicted == labels.float()).sum()
    loss = total_loss / len(dataloader)
    acc = correct_preds / (len(dataloader) * batch_size * num_classes)
    return loss, acc


if __name__ == "__main__":

    # Set parsable global variables.
    parser = argparse.ArgumentParser(description='LSTM variables')
    parser.add_argument('--embedding_dim', type=int, default=100,
                        help='Size of the word embeddings')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size for LSTM')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in LSTM')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    args = parser.parse_args()
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    epochs = args.epochs

    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device available:", device)

    # Loading data.
    products = load_product_data('inputs/products.json')
    products_test = load_product_data('inputs/products.test.json')
    embeddings = load_embeddings('inputs/glove.6B.100d.txt')

    # Create
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform([product['category'] for product in products])

    num_classes = len(mlb.classes_)

    # Create padded embedding for training data and pad sequence.
    max_len = max([len(product['name'].split()) for product in products if product['name'] is not None])
    train_embedding = [sentence_embedding(product['name'], embeddings, embedding_dim, max_len) for product in products]
    pad_sequence = nn.utils.rnn.pad_sequence(train_embedding, batch_first=True)

    # Get data loaders for train and validation set.
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    train_dataset = TensorDataset(pad_sequence, train_labels)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Move train and validation data to GPU if available.
    train_loader = [(to_device(inputs, device), to_device(labels, device)) for inputs, labels in train_loader]
    val_loader = [(to_device(inputs, device), to_device(labels, device)) for inputs, labels in val_loader]

    # Fetch model, loss and optimer.
    model = LSTM(embedding_dim, num_classes, hidden_dim, num_layers)
    model.to(device)  # Move model to GPU if available
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Lists to store metrics for plotting.
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training
    print("[*] Training model...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        correct_preds = 0
        for inputs, labels in train_loader:
            inputs, labels = to_device(
                inputs, device), to_device(labels, device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = loss_fn(outputs, labels.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = outputs.sigmoid() > 0.5
            correct_preds += (predicted == labels.float()).sum()

        # Compute metrics and store for plotting
        train_loss = total_loss / len(train_loader)
        train_accuracy = correct_preds / \
            (len(train_loader) * batch_size * num_classes)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        print("[*] Validating model...")
        val_loss, val_accuracy = test_model(model, val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"train Loss: {train_loss}, train acc: {train_accuracy}")
        print(f"val loss: {val_loss}, val acc: {val_accuracy}")

    # Fit equal mlb for testset (this disregards extra catergories that are in the test set).
    test_mlb = mlb.transform([product['category'] for product in products_test])
    test_embedding = [sentence_embedding(product['name'], embeddings, embedding_dim, max_len) for product in products_test]
    pad_sequence = nn.utils.rnn.pad_sequence(test_embedding, batch_first=True)
    test_labels = torch.tensor(test_mlb, dtype=torch.float32)
    test_dataset = TensorDataset(pad_sequence, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Move test data to GPU if available.
    test_loader = [(to_device(inputs, device), to_device(labels, device)) for inputs, labels in test_loader]

    # Test phase
    print("[*] Testing model...")
    test_loss, test_accuracy = test_model(model, test_loader)
    print(f"test loss: {test_loss}, test acc: {test_accuracy}")

    # Plotting
    epochs = range(1, epochs + 1)
    plt.figure(figsize=(10, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train')
    plt.plot(epochs, val_losses, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [acc.cpu() for acc in train_accuracies], label='Train')
    plt.plot(epochs, [acc.cpu() for acc in val_accuracies], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
