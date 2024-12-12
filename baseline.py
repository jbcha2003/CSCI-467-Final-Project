from collections import Counter
from sklearn.metrics import accuracy_score


def evaluate_baseline(y_train, y_test):
    # get the most frequent class
    majority_class = Counter(y_train).most_common(1)[0][0]

    # make the most frequent class as the prediction
    y_baseline_pred = [majority_class] * len(y_test)

    print("Baseline Method:")
    print("Accuracy:", accuracy_score(y_test, y_baseline_pred))
    return y_baseline_pred
