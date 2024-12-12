from data import preprocess_data, get_split_data
from baseline import evaluate_baseline
from log_reg import evaluate_logreg
from svm import evaluate_svm
from neuralnet import train_and_evaluate_nn
from plot import (
    print_labels,
    plot_svm_hyperparams,
    plot_nn_hyperparams,
    plot_misclassification,
)


def main():
    file_path = "Filtered_PTID_Data.csv"
    X, y = preprocess_data(file_path)
    X_train, X_dev, X_test, y_train, y_dev, y_test = get_split_data(X, y)

    # print distribution
    print_labels(X_train, y_train, X_dev, y_dev, X_test, y_test)

    # Baseline Test
    evaluate_baseline(y_train, y_test)

    # Logisitic Regression
    logistic_preds = evaluate_logreg(X_train, y_train, X_dev, y_dev)

    # SVM
    svm_preds, randoms_search_svm = evaluate_svm(
        X_train, y_train, X_dev, y_dev, X_test, y_test
    )
    plot_svm_hyperparams(randoms_search_svm)  # plot svm hyperparms

    # Neural Network
    nn_res, nn_preds = train_and_evaluate_nn(
        X_train, y_train, X_dev, y_dev, X_test, y_test
    )
    plot_nn_hyperparams(nn_res)  # plot nn hyperparams

    # Plot Misclassification
    plot_misclassification(y_test, logistic_preds, svm_preds, nn_preds)


if __name__ == "__main__":
    main()
