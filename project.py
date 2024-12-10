import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from collections import Counter
from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


def plot_hyperparameter_tuning_results(search_results, title):
    scores = search_results.cv_results_["mean_test_score"]
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(scores)), scores, marker="o", linestyle="-")
    plt.title(title)
    plt.xlabel("Hyperparameter Combination Index")
    plt.ylabel("Mean Cross-Validation Accuracy")
    plt.grid(True)
    plt.show()


def print_label_distribution(split_name, labels):
    label_counts = Counter(labels)
    print(f"\nLabel distribution in {split_name} set:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} examples")


def preprocess_data(file_path):
    # get input data
    data = pd.read_csv(file_path)

    # remove patient id and extraneous columns including form headers
    columns = [
        "Unnamed: 0",
        "PTID",
        "date_visit",
        "NACCVNUM",  # visit num
        "MMSELAN",  # language of test admin
    ]

    data_cleaned = data.drop(columns=columns)

    # replace -4 with na
    data_cleaned = data_cleaned.replace(-4, pd.NA)
    data_cleaned.dropna()

    # map categorical outcome with numeric
    data_cleaned["label"] = data_cleaned["label"].map({"low": 0, "inter": 1, "high": 2})

    # create X and y sets
    X = data_cleaned.drop(
        columns=[
            "label",
        ]
    ).apply(pd.to_numeric, errors="coerce")
    y = data_cleaned["label"]

    # impute to prevent errors with missing features
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # use scaler for efficiency
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # choose 10 best features
    sk = SelectKBest(f_classif, k=10)
    X_new = sk.fit_transform(X_scaled, y)
    labels = sk.get_support()

    X_new_df = pd.DataFrame(X_new, columns=X.columns[labels])

    print(X_new_df.head())

    return X_new_df, y


def split_data(X, y):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=42,
        stratify=y_train_full,
    )
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def evaluate_baseline(y_train, y_test):
    majority_class = Counter(y_train).most_common(1)[0][0]
    y_baseline_pred = [majority_class] * len(y_test)
    print("Baseline Method:")
    print("Accuracy:", accuracy_score(y_test, y_baseline_pred))
    return y_baseline_pred


def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(
        max_iter=5000, solver="saga", multi_class="multinomial", random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nLogistic Regression Model:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return model, y_pred


def train_svm_with_random_search(X_dev, y_dev):
    param_dist_svm = {
        "C": uniform(loc=0.1, scale=10),
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "gamma": ["scale", "auto"],
    }
    svm = SVC(random_state=42)
    random_search_svm = RandomizedSearchCV(
        estimator=svm,
        param_distributions=param_dist_svm,
        n_iter=20,
        scoring="accuracy",
        cv=5,
        verbose=2,
        n_jobs=-1,
        random_state=42,
    )
    random_search_svm.fit(X_dev, y_dev)
    print(
        "Best Parameters from Randomized Search for SVM:",
        random_search_svm.best_params_,
    )
    print(
        "Best Cross-Validation Accuracy from Randomized Search for SVM:",
        random_search_svm.best_score_,
    )
    plot_hyperparameter_tuning_results(random_search_svm, "SVM Hyperparameter Tuning")
    return random_search_svm.best_estimator_


def train_neural_network(
    X_train,
    y_train,
    X_dev,
    y_dev,
    input_size,
    hidden_size,
    output_size,
    epochs=50,
    batch_size=32,
):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    model = NeuralNetwork(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    return model


def train_and_evaluate_nn(
    X_train, y_train, X_dev, y_dev, input_size, output_size, param_grid
):
    best_model = None
    best_accuracy = 0
    best_params = None

    results = []

    for params in product(*param_grid.values()):
        hidden_size, lr, batch_size, epochs = params

        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        X_dev_tensor = torch.tensor(X_dev.values, dtype=torch.float32)
        y_dev_tensor = torch.tensor(y_dev.values, dtype=torch.long)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=batch_size,
            shuffle=True,
        )
        dev_loader = DataLoader(
            TensorDataset(X_dev_tensor, y_dev_tensor),
            batch_size=batch_size,
            shuffle=False,
        )

        model = NeuralNetwork(input_size, hidden_size, output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        y_dev_pred = []
        with torch.no_grad():
            for X_batch, _ in dev_loader:
                y_logits = model(X_batch)
                y_dev_pred.extend(torch.argmax(y_logits, dim=1).numpy())

        dev_accuracy = accuracy_score(y_dev, y_dev_pred)
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
        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            best_model = model
            best_params = params

    print(f"\nBest Parameters: {best_params}, Best Dev Accuracy: {best_accuracy:.4f}")
    return best_model, best_params, results


def evaluate_model(model, X_test, y_test):
    model.eval()

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False
    )

    y_pred = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            y_logits = model(X_batch)
            y_pred.extend(torch.argmax(y_logits, dim=1).numpy())

    acc = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy: {acc:.4f}")

    conf_matrix = confusion_matrix(y_test, y_pred)

    misclassified_per_class = {}
    for i in range(len(conf_matrix)):
        misclassified_count = sum(conf_matrix[i]) - conf_matrix[i][i]
        misclassified_per_class[f"True Label {i}"] = misclassified_count

    return y_pred, conf_matrix, misclassified_per_class


def plot_misclassification_errors(misclassified_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.2
    x = range(len(misclassified_df))
    ax.bar(
        x,
        misclassified_df["Logistic Regression"],
        width=bar_width,
        label="Logistic Regression",
    )
    ax.bar(
        [i + bar_width for i in x],
        misclassified_df["SVM"],
        width=bar_width,
        label="SVM",
    )
    ax.bar(
        [i + 2 * bar_width for i in x],
        misclassified_df["Neural Network"],
        width=bar_width,
        label="Neural Network",
    )
    ax.set_xticks([i + bar_width for i in x])
    label_mapping = {0: "low", 1: "med", 2: "high"}
    ax.set_xticklabels(
        [label_mapping[int(label[-1])] for label in misclassified_df["True Label"]]
    )
    ax.set_ylabel("Number of Misclassified Examples")
    ax.set_title("Misclassification Errors by Label for Each Model")
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_nn_tuning_results(results):
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=range(len(results_df)),
        y="dev_accuracy",
        hue="hidden_size",
        size="batch_size",
        palette="viridis",
        data=results_df,
        legend="full",
    )

    plt.title("Neural Network Hyperparameter Tuning")
    plt.xlabel("Hyperparameter Combination Index")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.legend(title="Hidden Size")
    plt.show()


def main():
    file_path = "Filtered_PTID_Data.csv"
    X, y = preprocess_data(file_path)
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(X, y)

    print(f"Training set size: {len(X_train)}")
    print(f"Development set size: {len(X_dev)}")
    print(f"Testing set size: {len(X_test)}")
    print_label_distribution("Training", y_train)
    print_label_distribution("Development", y_dev)
    print_label_distribution("Testing", y_test)

    # Baseline Test
    evaluate_baseline(y_train, y_test)

    # Logisitic Regression
    logistic_model, logistic_preds = train_logistic_regression(
        X_train, y_train, X_test, y_test
    )

    # SVM
    best_svm = train_svm_with_random_search(X_dev, y_dev)
    svm_preds = best_svm.predict(X_test)

    input_size = X_train.shape[1]
    output_size = len(y.unique())

    param_grid = {
        "hidden_size": [32, 64, 128],
        "lr": [0.001, 0.01, 0.1],
        "batch_size": [16, 32, 64],
        "epochs": [10, 20],
    }

    # Neural Network
    best_nn_model, best_nn_params, nn_results = train_and_evaluate_nn(
        X_train, y_train, X_dev, y_dev, input_size, output_size, param_grid
    )

    plot_nn_tuning_results(nn_results)

    nn_preds, nn_conf_matrix, nn_misclassified = evaluate_model(
        best_nn_model, X_test, y_test
    )

    # Print Misclassification Graph
    logistic_conf_matrix = confusion_matrix(y_test, logistic_preds)
    svm_conf_matrix = confusion_matrix(y_test, svm_preds)

    misclassified_data = {
        "True Label": [f"True Label {i}" for i in range(len(logistic_conf_matrix))],
        "Logistic Regression": [
            sum(logistic_conf_matrix[i]) - logistic_conf_matrix[i][i]
            for i in range(len(logistic_conf_matrix))
        ],
        "SVM": [
            sum(svm_conf_matrix[i]) - svm_conf_matrix[i][i]
            for i in range(len(svm_conf_matrix))
        ],
        "Neural Network": [
            nn_misclassified[f"True Label {i}"] for i in range(len(nn_conf_matrix))
        ],
    }

    misclassified_df = pd.DataFrame(misclassified_data)
    plot_misclassification_errors(misclassified_df)


if __name__ == "__main__":
    main()
