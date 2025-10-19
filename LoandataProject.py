# -----------------------------------------------
# PROJECT: Machine Learning Model on Loan Dataset
# MODEL: Logistic Regression
# AUTHOR: Nithish M
# -----------------------------------------------

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

# Step 2: Load the Dataset
# Make sure loan_data_set.csv is in the same folder as this script
df = pd.read_csv("D:\Data Science Project\ML model Application on Loan dataset/loan_data_set.csv")

print(" Dataset Loaded Successfully!")
print("Shape of Dataset:", df.shape)
print("\nFirst 5 Rows:\n", df.head())

# Step 3: Data Cleaning
print("\nChecking for Missing Values:")
print(df.isnull().sum())

# Drop duplicate rows if any
df = df.drop_duplicates()

# Fill missing numeric values with median
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\n Missing values handled successfully!")

# Step 4: Encode Categorical Columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("\nLabel Encoding done for categorical columns!")

# Step 5: Define Features and Target
# Change 'Loan_Status' to the correct target column name in your dataset if different
target_col = 'Loan_Status'
if target_col not in df.columns:
    raise ValueError(f"⚠️ Target column '{target_col}' not found. Check dataset column names!")

X = df.drop(target_col, axis=1)
y = df[target_col]

# Step 6: Split Data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTrain-Test Split Completed")
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Step 7: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("\nFeature Scaling Done")

# Step 8: Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("\n Model Training Completed")

# Step 9: Make Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Step 10: Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n Model Evaluation Metrics:")
print("Accuracy:", round(accuracy, 3))
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)
print("ROC-AUC Score:", round(roc_auc, 3))

# Step 11: Visualization

# Confusion Matrix Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()

# Step 12: Save Model and Results
import pickle
pickle.dump(model, open("loan_logistic_model.pkl", "wb"))

df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_results.to_csv("loan_predictions.csv", index=False)

print("\n Model and Predictions Saved Successfully!")
print("Model File: loan_logistic_model.pkl")
print("Predictions File: loan_predictions.csv")

print("\n Project Completed Successfully — Machine Learning Model on Loan Dataset!")
