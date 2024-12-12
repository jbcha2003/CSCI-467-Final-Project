from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import uniform


def evaluate_svm(X_train, y_train, X_dev, y_dev, X_test, y_test):
    RANDOM_STATE = 42

    # train initial model
    svm = SVC(random_state=RANDOM_STATE)
    svm.fit(X_train, y_train)
    basic_pred = svm.predict(X_test)
    print("Basic SVM Model:")
    print(
        "Test Accuracy without hyperparam selection:",
        accuracy_score(y_test, basic_pred),
    )
    print("\n")

    # hyperparamters
    param_dist_svm = {
        "C": uniform(loc=0.1, scale=10),
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "gamma": ["scale", "auto"],
    }

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
        "\nBest Parameters from Randomized Search for SVM:",
        random_search_svm.best_params_,
    )
    print(
        "Best Cross-Validation Accuracy from Randomized Search for SVM:",
        random_search_svm.best_score_,
    )

    preds = random_search_svm.predict(X_test)

    # evaluate on test set
    print("Test Accuracy with Hyperparameters:", accuracy_score(y_test, preds))
    print("\n")

    return preds, random_search_svm
