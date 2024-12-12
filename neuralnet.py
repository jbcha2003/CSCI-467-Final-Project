import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
from sklearn.metrics import accuracy_score, confusion_matrix


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


def evaluate_nn(model, X_test, y_test, batch_size):
    X_test_tn = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tn = torch.tensor(y_test.values, dtype=torch.long)

    test_loader = DataLoader(
        TensorDataset(X_test_tn, y_test_tn),
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()
    y_test_pred = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            y_logits = model(X_batch)
            y_test_pred.extend(torch.argmax(y_logits, dim=1).numpy())

    return accuracy_score(y_test, y_test_pred), y_test_pred


def train_nn(
    X_train,
    y_train,
    learning_rate,
    batch_size,
    input_size,
    hidden_size,
    output_size,
    epochs,
):
    X_train_tn = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tn = torch.tensor(y_train.values, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_tn, y_train_tn),
        batch_size=batch_size,
        shuffle=True,
    )

    model = NeuralNetwork(input_size, hidden_size, output_size)
    crit = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = crit(y_pred, y_batch)
            loss.backward()
            optimizer.step()

    return model


def train_and_evaluate_nn(X_train, y_train, X_dev, y_dev, X_test, y_test):
    best_model = None
    best_accuracy = 0
    best_params = None
    results = []

    input_size = X_train.shape[1]
    output_size = 3

    param_grid = {
        "hidden_size": [32, 64, 128],
        "lr": [0.001, 0.01, 0.1],
        "batch_size": [16, 32, 64],
        "epochs": [10, 20],
    }

    print("Neural Network: ")
    # select each set of hyperparmaters
    for params in product(*param_grid.values()):
        hidden_size, lr, batch_size, epochs = params

        # train model
        model = train_nn(
            X_train,
            y_train,
            lr,
            batch_size,
            input_size,
            hidden_size,
            output_size,
            epochs,
        )

        # evaluate model
        dev_accuracy, _ = evaluate_nn(model, X_dev, y_dev, batch_size)

        # add result
        results.append(
            {
                "hidden_size": hidden_size,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "dev_accuracy": dev_accuracy,
            }
        )

        print(f"Params: {params}, Dev Accuracy: {dev_accuracy:.4f}")

        # determine best accuracy
        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            best_model = model
            best_params = params

    print(f"\nBest Parameters: {best_params}, Best Dev Accuracy: {best_accuracy:.4f}")

    # evaluate on test data
    test_score, preds = evaluate_nn(best_model, X_test, y_test, batch_size)
    print("Best Test Score:", test_score)

    return results, preds
