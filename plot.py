import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter


def print_label_distribution(split_name, labels):
    label_counts = Counter(labels)
    print(f"\nLabel distribution in {split_name} set:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} examples")


def print_labels(X_train, y_train, X_dev, y_dev, X_test, y_test):
    print(f"Training set size: {len(X_train)}")
    print(f"Development set size: {len(X_dev)}")
    print(f"Testing set size: {len(X_test)}")
    print_label_distribution("Training", y_train)
    print_label_distribution("Development", y_dev)
    print_label_distribution("Testing", y_test)


def plot_svm_hyperparams(search_results):
    scores = search_results.cv_results_["mean_test_score"]
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(scores)), scores, marker="o", linestyle="-")
    plt.title("SVM Hyperparameters")
    plt.xlabel("Hyperparameter Combination Index")
    plt.ylabel("Mean Cross-Validation Accuracy")
    plt.grid(True)
    plt.show()


def plot_nn_hyperparams(results):
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


def plot_misclassification(y_true, logreg_preds, svm_preds, nn_preds):
    # Print Misclassification Graph
    logistic_conf_matrix = confusion_matrix(y_true, logreg_preds)
    svm_conf_matrix = confusion_matrix(y_true, svm_preds)
    nn_conf_matrix = confusion_matrix(y_true, nn_preds)

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
            sum(nn_conf_matrix[i]) - nn_conf_matrix[i][i]
            for i in range(len(nn_conf_matrix))
        ],
    }

    misclassified_df = pd.DataFrame(misclassified_data)

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
