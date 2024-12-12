from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def evaluate_logreg(X_train, y_train, X_test, y_test):
    # params
    MAX_ITER = 5000
    solver = "saga"
    multi_class = "multinomial"
    random_state = 42

    # define model
    model = LogisticRegression(
        max_iter=MAX_ITER,
        solver=solver,
        multi_class=multi_class,
        random_state=random_state,
    )

    # train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # print results
    print("\nLogistic Regression Model:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\n")

    return y_pred
