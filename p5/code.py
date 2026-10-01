# ============================================================
# SVM CLASSIFICATION - LOAN APPROVAL DATASET
# ============================================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, RFE

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("loan-approval.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 4. REMOVE ID COLUMN
# ============================================================

df = df.drop("ApplicantID", axis=1)


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("LoanApproved", axis=1)

# Convert target:
# N = 0
# Y = 1
y = df["LoanApproved"].map({"N": 0, "Y": 1})

print("\nTarget Distribution:")
print(y.value_counts())


# ============================================================
# 6. DEFINE NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = [
    "Age",
    "AnnualIncome",
    "LoanAmount",
    "CreditScore",
    "EmploymentYears"
]

categorical_features = [
    "Education",
    "MaritalStatus",
    "PropertyArea",
    "SelfEmployed"
]


# ============================================================
# 7. SCALE + ENCODE DATA
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),

        ("cat",
         OneHotEncoder(
             handle_unknown="ignore",
             sparse_output=False
         ),
         categorical_features)
    ]
)


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 9. SVM CLASSIFICATION
# ============================================================

svm_model = Pipeline([
    ("preprocessing", preprocessor),

    ("svm",
     SVC(
         kernel="rbf",
         C=1.0,
         gamma="scale",
         random_state=42
     ))
])


# Train model
svm_model.fit(X_train, y_train)


# ============================================================
# 10. PREDICTION
# ============================================================

y_pred = svm_model.predict(X_test)

# Decision scores for AUC
y_score = svm_model.decision_function(X_test)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

print("\n================ CONFUSION MATRIX ================")
print(cm)

print("\nTrue Negative (TN):", TN)
print("False Positive (FP):", FP)
print("False Negative (FN):", FN)
print("True Positive (TP):", TP)


# ============================================================
# 12. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

error = 1 - accuracy

recall = recall_score(y_test, y_pred)

specificity = TN / (TN + FP)

precision = precision_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

auc = roc_auc_score(y_test, y_score)


print("\n================ SVM RESULTS ================")

print("Accuracy     :", round(accuracy, 4))
print("Error        :", round(error, 4))
print("Recall       :", round(recall, 4))
print("Specificity  :", round(specificity, 4))
print("Precision    :", round(precision, 4))
print("F1-Score     :", round(f1, 4))
print("AUC          :", round(auc, 4))


# ============================================================
# 13. 5-FOLD CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    svm_model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\n================ 5-FOLD CROSS VALIDATION ================")

print("Fold 1 Accuracy:", round(cv_scores[0], 4))
print("Fold 2 Accuracy:", round(cv_scores[1], 4))
print("Fold 3 Accuracy:", round(cv_scores[2], 4))
print("Fold 4 Accuracy:", round(cv_scores[3], 4))
print("Fold 5 Accuracy:", round(cv_scores[4], 4))

print("\nMean CV Accuracy:",
      round(cv_scores.mean(), 4))

print("CV Standard Deviation:",
      round(cv_scores.std(), 4))


# ============================================================
# 14. PREPARE DATA FOR FEATURE SELECTION
# ============================================================

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

# Get feature names after encoding
feature_names = preprocessor.get_feature_names_out()

print("\nTotal Features After Encoding:",
      len(feature_names))


# ============================================================
# 15. FEATURE SELECTION - SELECTKBEST
# ============================================================

k = min(8, X_train_processed.shape[1])

selector = SelectKBest(
    score_func=f_classif,
    k=k
)

X_train_kbest = selector.fit_transform(
    X_train_processed,
    y_train
)

X_test_kbest = selector.transform(
    X_test_processed
)

selected_features = feature_names[
    selector.get_support()
]

print("\n================ SELECTKBEST ================")

print("Selected Features:")

for feature in selected_features:
    print(feature)


# ============================================================
# 16. SVM WITH SELECTKBEST FEATURES
# ============================================================

svm_kbest = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    random_state=42
)

svm_kbest.fit(
    X_train_kbest,
    y_train
)

y_pred_kbest = svm_kbest.predict(
    X_test_kbest
)

y_score_kbest = svm_kbest.decision_function(
    X_test_kbest
)


# ============================================================
# 17. SELECTKBEST METRICS
# ============================================================

cm_kbest = confusion_matrix(
    y_test,
    y_pred_kbest
)

TN_k, FP_k, FN_k, TP_k = cm_kbest.ravel()

accuracy_k = accuracy_score(
    y_test,
    y_pred_kbest
)

error_k = 1 - accuracy_k

recall_k = recall_score(
    y_test,
    y_pred_kbest
)

specificity_k = TN_k / (TN_k + FP_k)

f1_k = f1_score(
    y_test,
    y_pred_kbest
)

auc_k = roc_auc_score(
    y_test,
    y_score_kbest
)


print("\nConfusion Matrix:")
print(cm_kbest)

print("\nTP:", TP_k)
print("TN:", TN_k)
print("FP:", FP_k)
print("FN:", FN_k)

print("\nAccuracy:", round(accuracy_k, 4))
print("Error:", round(error_k, 4))
print("Recall:", round(recall_k, 4))
print("Specificity:", round(specificity_k, 4))
print("F1-Score:", round(f1_k, 4))
print("AUC:", round(auc_k, 4))


# ============================================================
# 18. FEATURE SELECTION - RFE
# ============================================================

rfe_estimator = SVC(
    kernel="linear"
)

n_features = X_train_processed.shape[1]

n_select = max(
    1,
    n_features // 2
)

rfe = RFE(
    estimator=rfe_estimator,
    n_features_to_select=n_select,
    step=1
)

X_train_rfe = rfe.fit_transform(
    X_train_processed,
    y_train
)

X_test_rfe = rfe.transform(
    X_test_processed
)

rfe_features = feature_names[
    rfe.support_
]


print("\n================ RFE FEATURE SELECTION ================")

print("Selected Features:")

for feature in rfe_features:
    print(feature)


# ============================================================
# 19. SVM WITH RFE FEATURES
# ============================================================

svm_rfe = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    random_state=42
)

svm_rfe.fit(
    X_train_rfe,
    y_train
)

y_pred_rfe = svm_rfe.predict(
    X_test_rfe
)

y_score_rfe = svm_rfe.decision_function(
    X_test_rfe
)


# ============================================================
# 20. RFE METRICS
# ============================================================

cm_rfe = confusion_matrix(
    y_test,
    y_pred_rfe
)

TN_r, FP_r, FN_r, TP_r = cm_rfe.ravel()

accuracy_r = accuracy_score(
    y_test,
    y_pred_rfe
)

error_r = 1 - accuracy_r

recall_r = recall_score(
    y_test,
    y_pred_rfe
)

specificity_r = TN_r / (TN_r + FP_r)

f1_r = f1_score(
    y_test,
    y_pred_rfe
)

auc_r = roc_auc_score(
    y_test,
    y_score_rfe
)


print("\nConfusion Matrix:")
print(cm_rfe)

print("\nTP:", TP_r)
print("TN:", TN_r)
print("FP:", FP_r)
print("FN:", FN_r)

print("\nAccuracy:", round(accuracy_r, 4))
print("Error:", round(error_r, 4))
print("Recall:", round(recall_r, 4))
print("Specificity:", round(specificity_r, 4))
print("F1-Score:", round(f1_r, 4))
print("AUC:", round(auc_r, 4))


# ============================================================
# 21. FINAL COMPARISON
# ============================================================

results = pd.DataFrame({

    "Method": [
        "SVM - All Features",
        "SVM - SelectKBest",
        "SVM - RFE"
    ],

    "Accuracy": [
        accuracy,
        accuracy_k,
        accuracy_r
    ],

    "Error": [
        error,
        error_k,
        error_r
    ],

    "Recall": [
        recall,
        recall_k,
        recall_r
    ],

    "Specificity": [
        specificity,
        specificity_k,
        specificity_r
    ],

    "F1-Score": [
        f1,
        f1_k,
        f1_r
    ],

    "AUC": [
        auc,
        auc_k,
        auc_r
    ]
})


print("\n================ FINAL COMPARISON ================")

print(
    results.round(4).to_string(index=False)
)