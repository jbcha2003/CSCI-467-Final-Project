import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from collections import Counter
from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Define plotting function for hyperparameter tuning results
def plot_hyperparameter_tuning_results(search_results, title):
    # Extract the cross-validation scores and parameter settings
    scores = search_results.cv_results_['mean_test_score']
    params = search_results.cv_results_['params']
    
    # Plot the scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(scores)), scores, marker='o', linestyle='-')
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

# Split the dataset into training, development, and testing sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Further split the training set into training and dev (validation) sets
X_train, X_dev, y_train, y_dev = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)

# Print the sizes of each split
print(f"Training set size: {len(X_train)}")
print(f"Development set size: {len(X_dev)}")
print(f"Testing set size: {len(X_test)}")

print_label_distribution("Training", y_train)
print_label_distribution("Development", y_dev)
print_label_distribution("Testing", y_test)

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
param_dist_svm = {
    'C': uniform(loc=0.1, scale=10),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

# Initialize SVM model
svm = SVC(random_state=42)

# Set up RandomizedSearchCV for SVM with 5-fold cross-validation, using the dev set
random_search_svm = RandomizedSearchCV(
    estimator=svm,
    param_distributions=param_dist_svm,
    n_iter=20,
    scoring='accuracy',
    cv=5,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Fit RandomizedSearchCV to the training data, validating on the dev set
random_search_svm.fit(X_dev, y_dev)

# Display best parameters and best score from randomized search
print("Best Parameters from Randomized Search for SVM:", random_search_svm.best_params_)
print("Best Cross-Validation Accuracy from Randomized Search for SVM:", random_search_svm.best_score_)

# Plot hyperparameter tuning results for SVM
plot_hyperparameter_tuning_results(random_search_svm, "SVM Hyperparameter Tuning")

# Additional Step: Calculate Misclassifications for SVM
# Predict using the best SVM model
y_svm_pred = random_search_svm.best_estimator_.predict(X_test)

# Calculate confusion matrix for SVM predictions
conf_matrix_svm = confusion_matrix(y_test, y_svm_pred)
misclassified_per_class_svm = {}

for i in range(len(conf_matrix_svm)):
    misclassified_count = sum(conf_matrix_svm[i]) - conf_matrix_svm[i][i]
    misclassified_per_class_svm[f"True Label {i}"] = misclassified_count

# Combine misclassification data from each model
misclass_data = {
    'True Label': [f'True Label {i}' for i in range(len(conf_matrix))],
    'Logistic Regression': [misclassified_per_class[f'True Label {i}'] for i in range(len(conf_matrix))],
    'SVM': [misclassified_per_class_svm[f'True Label {i}'] for i in range(len(conf_matrix))]
}

misclass_df = pd.DataFrame(misclass_data)

# Plot the misclassification errors by label for each model
fig, ax = plt.subplots(figsize=(10, 6))

# Set positions for each model's bars
x = range(len(misclass_df))
bar_width = 0.25  # Width of each bar

# Plot each model's misclassifications
ax.bar([i + bar_width for i in x], misclass_df['Logistic Regression'], width=bar_width, label='Logistic Regression')
ax.bar([i + 2 * bar_width for i in x], misclass_df['SVM'], width=bar_width, label='SVM')

# Customize the plot
ax.set_xticks([i + bar_width for i in x])
ax.set_xticklabels(misclass_df['True Label'])
ax.set_ylabel("Number of Misclassified Examples")
ax.set_title("Misclassification Errors by Label for Each Model")
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
