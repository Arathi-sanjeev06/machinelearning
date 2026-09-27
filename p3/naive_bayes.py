# ============================================================
# NAIVE BAYES CLASSIFICATION
# Loan Approval Dataset
# ============================================================

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    f1_score,
    roc_auc_score
)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("loan_approval.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

# ------------------------------------------------------------
# 2. REMOVE APPLICANT ID
# ------------------------------------------------------------

df = df.drop("ApplicantID", axis=1)

# ------------------------------------------------------------
# 3. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop("LoanApproved", axis=1)
y = df["LoanApproved"]

# ------------------------------------------------------------
# 4. HANDLE MISSING VALUES
# ------------------------------------------------------------

# Numerical columns
numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns

for column in numeric_columns:
    X[column] = X[column].fillna(X[column].median())

# Categorical columns
categorical_columns = X.select_dtypes(include=["str", "object"]).columns

for column in categorical_columns:
    X[column] = X[column].fillna(X[column].mode()[0])

# ------------------------------------------------------------
# 5. ENCODE CATEGORICAL FEATURES
# ------------------------------------------------------------

# Convert all categorical values into numerical columns
X = pd.get_dummies(X, dtype=float)

# ------------------------------------------------------------
# 6. ENCODE TARGET
# ------------------------------------------------------------

y = y.map({
    "N": 0,
    "Y": 1
})

print("\nClasses:")
print(["N", "Y"])

# ------------------------------------------------------------
# 7. CHECK DATA
# ------------------------------------------------------------

print("\nMissing values after preprocessing:")
print(X.isnull().sum().sum())

print("\nNumber of features after encoding:", X.shape[1])

# ------------------------------------------------------------
# 8. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 9. CREATE NAIVE BAYES MODEL
# ------------------------------------------------------------

model = GaussianNB()

# ------------------------------------------------------------
# 10. TRAIN MODEL
# ------------------------------------------------------------

model.fit(X_train, y_train)

# ------------------------------------------------------------
# 11. PREDICTION
# ------------------------------------------------------------

y_pred = model.predict(X_test)

# Probability for AUC
y_prob = model.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------
# 12. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

print("\n======================================")
print("CONFUSION MATRIX")
print("======================================")

print(cm)

print("\nTN =", TN)
print("FP =", FP)
print("FN =", FN)
print("TP =", TP)

# ------------------------------------------------------------
# 13. ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print("CLASSIFICATION METRICS")
print("======================================")

print("Accuracy =", accuracy)
print("Accuracy (%) =", accuracy * 100)

# ------------------------------------------------------------
# 14. ERROR
# ------------------------------------------------------------

error = 1 - accuracy

print("Error =", error)
print("Error (%) =", error * 100)

# ------------------------------------------------------------
# 15. RECALL / SENSITIVITY
# ------------------------------------------------------------

recall = recall_score(y_test, y_pred)

print("Recall =", recall)

# ------------------------------------------------------------
# 16. SPECIFICITY
# ------------------------------------------------------------

specificity = TN / (TN + FP)

print("Specificity =", specificity)

# ------------------------------------------------------------
# 17. F1 SCORE
# ------------------------------------------------------------

f1 = f1_score(y_test, y_pred)

print("F1-Score =", f1)

# ------------------------------------------------------------
# 18. AUC
# ------------------------------------------------------------

auc = roc_auc_score(y_test, y_prob)

print("AUC =", auc)

# ------------------------------------------------------------
# 19. K-FOLD CROSS VALIDATION
# ------------------------------------------------------------

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring="accuracy"
)

print("\n======================================")
print("5-FOLD CROSS VALIDATION")
print("======================================")

print("Fold Accuracies:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
