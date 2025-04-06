import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

logger.add("training.log", format="{message}")

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def load_data(train_csv, val_csv, test_csv):
    train_df = pd.read_csv(train_csv).drop(columns=['order1', 'order2'])
    val_df = pd.read_csv(val_csv).drop(columns=['order1', 'order2'])
    test_df = pd.read_csv(test_csv)

    X_train = train_df.drop(columns=['order0']).values
    y_train = train_df['order0'].values
    X_val = val_df.drop(columns=['order0']).values
    y_val = val_df['order0'].values
    X_test = test_df.values

    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    return X_train, y_train, X_val, y_val, X_test

def init_model(input_dim, lr):
    model = MLP(input_dim, hidden_size=256, output_size=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predictions = torch.max(outputs, 1)
        accuracy = accuracy_score(y, predictions)
        conf_matrix = confusion_matrix(y, predictions)
    return predictions, accuracy, conf_matrix

def train(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size):
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    best_val_accuracy = 0.0
    best_model = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        _, val_accuracy, _ = evaluate(model, X_val, y_val)
        logger.info(f'Эпоха {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()

    model.load_state_dict(best_model)
    logger.info(f"Лучшая точность: {best_val_accuracy:.4f}")
    return model

def main(args):
    X_train, y_train, X_val, y_val, X_test = load_data(args.train_csv, args.val_csv, args.test_csv)
    logger.info(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")

    input_dim = X_train.shape[1]
    model, criterion, optimizer = init_model(input_dim, args.lr)

    model = train(model, criterion, optimizer, X_train, y_train, X_val, y_val, args.num_epoches, args.batch_size)

    predictions, _, _ = evaluate(model, X_test, torch.zeros(X_test.size(0), dtype=torch.long))
    pd.DataFrame(predictions.numpy(), columns=['order0']).to_csv(args.out_csv, index=False)
    logger.info("Предсказания сохранены в " + args.out_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', default='data/train.csv')
    parser.add_argument('--val_csv', default='data/val.csv')
    parser.add_argument('--test_csv', default='data/test.csv')
    parser.add_argument('--out_csv', default='data/submission.csv')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epoches', type=int, default=10)

    args = parser.parse_args()
    main(args)
