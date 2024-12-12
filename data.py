import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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

    return X_new_df, y


def get_split_data(X, y):
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    DEV_SIZE = 0.25

    # split the data into train and test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # split the data into train and dev
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_full,
        y_train_full,
        test_size=DEV_SIZE,
        random_state=RANDOM_STATE,
    )
    return X_train, X_dev, X_test, y_train, y_dev, y_test
