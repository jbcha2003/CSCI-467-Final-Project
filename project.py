import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from collections import Counter
from scipy.stats import uniform  # Import uniform for randomized search
import numpy as np


# Load the data
data = pd.read_csv('Filtered_PTID_Data.csv')

# Drop unnecessary columns
data_cleaned = data.drop(columns=['Unnamed: 0', 'PTID'])

# Replace placeholder values (-4) with NaN for missing data handling
data_cleaned = data_cleaned.replace(-4, pd.NA)

# Convert the 'label' column to numerical categories
data_cleaned['label'] = data_cleaned['label'].map({'low': 0, 'inter': 1, 'high': 2})

# Separate features and target variable
X = data_cleaned.drop(columns=['label'])
y = data_cleaned['label']

# Convert all columns to numeric, coercing errors to NaN
X = X.apply(pd.to_numeric, errors='coerce')

# Impute missing values using the median strategy
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

### Baseline Method ###
# Predict the most common label (majority class) in the training set
majority_class = Counter(y_train).most_common(1)[0][0]
y_baseline_pred = [majority_class] * len(y_test)

# Evaluate Baseline
print("Baseline Method:")
print("Accuracy:", accuracy_score(y_test, y_baseline_pred))

### Logistic Regression Model ###
# Initialize and fit the logistic regression model
model = LogisticRegression(max_iter=5000, solver='saga', multi_class='multinomial', random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)

# Evaluate Logistic Regression
print("\nLogistic Regression Model:")
print("Accuracy:", accuracy_score(y_test, y_pred))

### Experimentation Summary ###
print("\nExperimentation Summary:")
print(f"Baseline Accuracy: {accuracy_score(y_test, y_baseline_pred):.2f}")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.2f}")

### Error Analysis ###
conf_matrix = confusion_matrix(y_test, y_pred)
misclassified_per_class = {}

for i in range(len(conf_matrix)):
    misclassified_count = sum(conf_matrix[i]) - conf_matrix[i][i]  # Total errors per true class
    misclassified_per_class[f"True Label {i}"] = misclassified_count

# Display which labels are most often misclassified
print("\nMisclassifications by True Label:")
for label, count in misclassified_per_class.items():
    print(f"{label}: {count} misclassified examples")


### Support Vector Machine (SVM) with Randomized Search for Hyperparameter Tuning ###
# Define parameter distribution for SVM
param_dist_svm = {
    'C': uniform(loc=0.1, scale=10),  # Regularization parameter in range [0.1, 10]
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Different kernel functions
    'gamma': ['scale', 'auto']  # Gamma settings for non-linear kernels
}

# Initialize SVM model
svm = SVC(random_state=42)

# Set up RandomizedSearchCV for SVM with 5-fold cross-validation
random_search_svm = RandomizedSearchCV(
    estimator=svm,
    param_distributions=param_dist_svm,
    n_iter=20,  # Number of random combinations to try
    scoring='accuracy',  # Metric to optimize
    cv=5,  # 5-fold cross-validation
    verbose=2,  # Print progress
    n_jobs=-1,  # Use all available cores
    random_state=42
)

# Fit RandomizedSearchCV to the data
random_search_svm.fit(X_train, y_train)

# Display best parameters and best score from randomized search
print("Best Parameters from Randomized Search for SVM:", random_search_svm.best_params_)
print("Best Cross-Validation Accuracy from Randomized Search for SVM:", random_search_svm.best_score_)

### Train the Final SVM Model with Best Parameters ###
best_svm = random_search_svm.best_estimator_
best_svm.fit(X_train, y_train)

# Predict and evaluate the final SVM model
y_svm_pred = best_svm.predict(X_test)

print("\nSupport Vector Machine Model with Best Hyperparameters:")
print("Accuracy:", accuracy_score(y_test, y_svm_pred))

### Error Analysis for SVM ###
# Count total number of misclassified examples
total_misclassified_svm = np.sum(y_test != y_svm_pred)
print("\nError Analysis for SVM:")
print(f"Total Misclassified Examples: {total_misclassified_svm}")

# Analyze which labels are most often misclassified
conf_matrix_svm = confusion_matrix(y_test, y_svm_pred)
misclassified_per_class_svm = {}

for i in range(len(conf_matrix_svm)):
    misclassified_count_svm = sum(conf_matrix_svm[i]) - conf_matrix_svm[i][i]  # Total errors per true class
    misclassified_per_class_svm[f"True Label {i}"] = misclassified_count_svm

# Display which labels are most often misclassified for SVM
print("\nMisclassifications by True Label for SVM:")
for label, count in misclassified_per_class_svm.items():
    print(f"{label}: {count} misclassified examples")
