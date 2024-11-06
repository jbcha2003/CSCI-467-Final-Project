import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

data = pd.read_csv('Filtered_PTID_Data.csv')

data_cleaned = data.drop(columns=['Unnamed: 0', 'PTID'])

# Replace placeholder values (-4) with NaN for missing data handling
data_cleaned = data_cleaned.applymap(lambda x: pd.NA if x == -4 else x)

# Convert the 'label' column to numerical categories
data_cleaned['label'] = data_cleaned['label'].map({'low': 0, 'inter': 1, 'high': 2})

# Separate features and target variable
X = data_cleaned.drop(columns=['label'])
y = data_cleaned['label']

# Ensure all columns are numeric by converting object types (e.g., dates) to numeric or dropping them
X = X.apply(pd.to_numeric, errors='coerce')

# Impute missing values using the median strategy
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize and fit the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
