# ============================================================
# K-NEAREST NEIGHBOUR (KNN) CLASSIFICATION
# Problem 2.2 - Why the choice of K matters
# ============================================================

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix


# ------------------------------------------------------------
# 1. CREATE DATASET
# ------------------------------------------------------------

data = {
    "x1": [4, 3, 5, 2, 8, 7],
    "x2": [3, 3, 5, 4, 8, 2],
    "Class": ["B", "A", "A", "A", "B", "B"]
}

df = pd.DataFrame(data)

print("======================================")
print("K-NEAREST NEIGHBOUR CLASSIFICATION")
print("======================================")

print("\nDataset:")
print(df)


# ------------------------------------------------------------
# 2. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

X = df[["x1", "x2"]]
y = df["Class"]


# ------------------------------------------------------------
# 3. ENCODE CLASS LABELS
# ------------------------------------------------------------

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nClasses:")
print(label_encoder.classes_)


# ------------------------------------------------------------
# 4. NEW POINT
# ------------------------------------------------------------

Q = np.array([[4, 4]])

print("\nNew Point Q =", Q[0])


# ------------------------------------------------------------
# 5. CALCULATE DISTANCES MANUALLY
# ------------------------------------------------------------

print("\n======================================")
print("DISTANCES FROM Q")
print("======================================")

distances = np.sqrt(
    (X["x1"].values - 4) ** 2 +
    (X["x2"].values - 4) ** 2
)

distance_table = df.copy()
distance_table["Distance"] = distances

distance_table = distance_table.sort_values(
    by="Distance"
).reset_index(drop=True)

print(distance_table)


# ------------------------------------------------------------
# 6. K = 1
# ------------------------------------------------------------

print("\n======================================")
print("K = 1")
print("======================================")

knn_1 = KNeighborsClassifier(
    n_neighbors=1,
    weights="uniform",
    algorithm="auto",
    leaf_size=30,
    p=2,
    metric="minkowski",
    metric_params=None,
    n_jobs=None
)

knn_1.fit(X, y_encoded)

prediction_1 = knn_1.predict(Q)

predicted_class_1 = label_encoder.inverse_transform(
    prediction_1
)

print("Predicted Class =", predicted_class_1[0])


# ------------------------------------------------------------
# 7. K = 3
# ------------------------------------------------------------

print("\n======================================")
print("K = 3")
print("======================================")

knn_3 = KNeighborsClassifier(
    n_neighbors=3,
    weights="uniform",
    algorithm="auto",
    leaf_size=30,
    p=2,
    metric="minkowski",
    metric_params=None,
    n_jobs=None
)

knn_3.fit(X, y_encoded)

prediction_3 = knn_3.predict(Q)

predicted_class_3 = label_encoder.inverse_transform(
    prediction_3
)

print("Predicted Class =", predicted_class_3[0])


# ------------------------------------------------------------
# 8. K = 5
# ------------------------------------------------------------

print("\n======================================")
print("K = 5")
print("======================================")

knn_5 = KNeighborsClassifier(
    n_neighbors=5,
    weights="uniform",
    algorithm="auto",
    leaf_size=30,
    p=2,
    metric="minkowski",
    metric_params=None,
    n_jobs=None
)

knn_5.fit(X, y_encoded)

prediction_5 = knn_5.predict(Q)

predicted_class_5 = label_encoder.inverse_transform(
    prediction_5
)

print("Predicted Class =", predicted_class_5[0])


# ------------------------------------------------------------
# 9. SHOW NEAREST NEIGHBOURS
# ------------------------------------------------------------

print("\n======================================")
print("NEAREST NEIGHBOURS")
print("======================================")

for k, model in [(1, knn_1), (3, knn_3), (5, knn_5)]:

    distances_k, indices_k = model.kneighbors(Q)

    print("\nK =", k)

    for i in range(k):
        index = indices_k[0][i]

        print(
            "Neighbour:",
            "P" + str(index + 1),
            "| x1 =", X.iloc[index]["x1"],
            "| x2 =", X.iloc[index]["x2"],
            "| Class =", y.iloc[index],
            "| Distance =", round(distances_k[0][i], 4)
        )


# ------------------------------------------------------------
# 10. FINAL COMPARISON
# ------------------------------------------------------------

print("\n======================================")
print("FINAL COMPARISON")
print("======================================")

print("K = 1  -->", predicted_class_1[0])
print("K = 3  -->", predicted_class_3[0])
print("K = 5  -->", predicted_class_5[0])


# ------------------------------------------------------------
# 11. IMPORTANT KNN PARAMETERS
# ------------------------------------------------------------

print("\n======================================")
print("KNN PARAMETERS USED")
print("======================================")

print("n_neighbors = K")
print("weights = uniform")
print("algorithm = auto")
print("leaf_size = 30")
print("p = 2")
print("metric = minkowski")
print("metric_params = None")
print("n_jobs = None")