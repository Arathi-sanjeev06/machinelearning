# ============================================================
# MAIZE PRODUCTION REGRESSION
# Linear Regression
# Polynomial Regression
# Multivariate Linear Regression
# ============================================================

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("maize.csv")

print("\nOriginal Dataset:")
print(df.head())

print("\nColumns in Dataset:")
print(df.columns.tolist())


# ============================================================
# 2. PREPARE MAIZE DATA
# ============================================================

# Convert FAOSTAT long-format data into columns
maize = df.pivot_table(
    index="Year",
    columns="Element",
    values="Value",
    aggfunc="first"
).reset_index()

# Keep only required columns
maize = maize[
    ["Year", "Area harvested", "Production", "Yield"]
]

# Remove missing values
maize = maize.dropna()

# Sort according to year
maize = maize.sort_values("Year").reset_index(drop=True)


print("\nColumns in Dataset:")
print(maize.columns.tolist())

print("\nDataset Size:")
print(maize.shape)

print("\nFirst Year:", maize["Year"].min())
print("Last Year :", maize["Year"].max())


# ============================================================
# 3. TRAINING AND TESTING DATA
# ============================================================

train = maize[maize["Year"] <= 2011].copy()
test = maize[maize["Year"] >= 2012].copy()

print("\n============================================")
print("TRAINING AND TESTING DATA")
print("============================================")

print(
    "Training Period:",
    train["Year"].min(),
    "-",
    train["Year"].max()
)

print(
    "Testing Period :",
    test["Year"].min(),
    "-",
    test["Year"].max()
)

print("Training Observations:", len(train))
print("Testing Observations:", len(test))


# ============================================================
# 4. LINEAR REGRESSION
# ============================================================

print("\n============================================")
print("LINEAR REGRESSION")
print("============================================")

X_train = train[["Year"]]
X_test = test[["Year"]]

y_train = train["Production"]
y_test = test["Production"]

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_prediction = linear_model.predict(X_test)

w = linear_model.coef_[0]
b = linear_model.intercept_

print("Hypothesis:")
print("h(w,b) = w*x + b")

print("\nw =", w)
print("b =", b)

print("\nEquation:")
print(
    "h(w,b) = ("
    + str(round(w, 4))
    + ")x + ("
    + str(round(b, 4))
    + ")"
)


# ============================================================
# 5. POLYNOMIAL REGRESSION
# ============================================================

print("\n============================================")
print("POLYNOMIAL REGRESSION")
print("============================================")

poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()

poly_model.fit(
    X_train_poly,
    y_train
)

poly_prediction = poly_model.predict(
    X_test_poly
)

w1 = poly_model.coef_[0]
w2 = poly_model.coef_[1]
b_poly = poly_model.intercept_

print("Hypothesis:")
print("h(w,b) = w2*x^2 + w1*x + b")

print("\nw1 =", w1)
print("w2 =", w2)
print("b  =", b_poly)

print("\nEquation:")
print(
    "h(w,b) = ("
    + str(round(w2, 8))
    + ")x^2 + ("
    + str(round(w1, 8))
    + ")x + ("
    + str(round(b_poly, 4))
    + ")"
)


# ============================================================
# 6. MULTIVARIATE LINEAR REGRESSION
# ============================================================

print("\n============================================")
print("MULTIVARIATE LINEAR REGRESSION")
print("============================================")

# x1 = Year
# x2 = Area harvested
# x3 = Yield
# y  = Production

features = [
    "Year",
    "Area harvested",
    "Yield"
]

X_multi_train = train[features]
X_multi_test = test[features]

multi_model = LinearRegression()

multi_model.fit(
    X_multi_train,
    y_train
)

multi_prediction = multi_model.predict(
    X_multi_test
)

w1_multi = multi_model.coef_[0]
w2_multi = multi_model.coef_[1]
w3_multi = multi_model.coef_[2]
b_multi = multi_model.intercept_

print("Hypothesis:")
print("h(w,b) = w1*x1 + w2*x2 + w3*x3 + b")

print("\nw1 =", w1_multi)
print("w2 =", w2_multi)
print("w3 =", w3_multi)
print("b  =", b_multi)

print("\nEquation:")
print(
    "h(w,b) = ("
    + str(round(w1_multi, 4))
    + ")x1 + ("
    + str(round(w2_multi, 6))
    + ")x2 + ("
    + str(round(w3_multi, 4))
    + ")x3 + ("
    + str(round(b_multi, 4))
    + ")"
)


# ============================================================
# 7. MODEL EVALUATION
# ============================================================

def evaluate_model(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, rmse, r2


linear_mae, linear_rmse, linear_r2 = evaluate_model(
    y_test,
    linear_prediction
)

poly_mae, poly_rmse, poly_r2 = evaluate_model(
    y_test,
    poly_prediction
)

multi_mae, multi_rmse, multi_r2 = evaluate_model(
    y_test,
    multi_prediction
)


# ============================================================
# 8. PRINT EVALUATION
# ============================================================

print("\nLinear Regression")
print("--------------------------------")
print("MAE  :", round(linear_mae, 2))
print("RMSE :", round(linear_rmse, 2))
print("R2   :", round(linear_r2, 4))


print("\nPolynomial Regression")
print("--------------------------------")
print("MAE  :", round(poly_mae, 2))
print("RMSE :", round(poly_rmse, 2))
print("R2   :", round(poly_r2, 4))


print("\nMultivariate Regression")
print("--------------------------------")
print("MAE  :", round(multi_mae, 2))
print("RMSE :", round(multi_rmse, 2))
print("R2   :", round(multi_r2, 4))


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Polynomial Regression",
        "Multivariate Regression"
    ],

    "MAE": [
        linear_mae,
        poly_mae,
        multi_mae
    ],

    "RMSE": [
        linear_rmse,
        poly_rmse,
        multi_rmse
    ],

    "R2": [
        linear_r2,
        poly_r2,
        multi_r2
    ]
})


print("\n============================================")
print("MODEL COMPARISON")
print("============================================")

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 10. FUTURE PRODUCTION PREDICTION
# ============================================================

future_years = pd.DataFrame({
    "Year": [2027, 2028, 2029, 2030]
})


# ---------- Linear Prediction ----------

future_linear = linear_model.predict(
    future_years[["Year"]]
)


# ---------- Polynomial Prediction ----------

future_poly = poly.transform(
    future_years[["Year"]]
)

future_polynomial = poly_model.predict(
    future_poly
)


# ---------- Multivariate Prediction ----------

# Use the latest available Area harvested and Yield
# from 2024 for future prediction.

latest_area = maize.iloc[-1]["Area harvested"]
latest_yield = maize.iloc[-1]["Yield"]

future_multi = pd.DataFrame({

    "Year": [2027, 2028, 2029, 2030],

    "Area harvested": [
        latest_area,
        latest_area,
        latest_area,
        latest_area
    ],

    "Yield": [
        latest_yield,
        latest_yield,
        latest_yield,
        latest_yield
    ]
})

future_multivariate = multi_model.predict(
    future_multi
)


# ============================================================
# 11. FUTURE PREDICTION TABLE
# ============================================================

future_results = pd.DataFrame({

    "Year": [2027, 2028, 2029, 2030],

    "Linear Prediction": future_linear,

    "Polynomial Prediction": future_polynomial,

    "Multivariate Prediction": future_multivariate
})


print("\n============================================")
print("      FUTURE MAIZE PRODUCTION PREDICTION")
print("              2027 - 2030")
print("============================================")

print(
    future_results.to_string(
        index=False,
        formatters={
            "Linear Prediction": "{:,.0f}".format,
            "Polynomial Prediction": "{:,.0f}".format,
            "Multivariate Prediction": "{:,.0f}".format
        }
    )
)
# ============================================================
# 12. COMBINED GRAPH
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))

# ------------------------------------------------------------
# ACTUAL MAIZE PRODUCTION
# ------------------------------------------------------------

plt.plot(
    maize["Year"],
    maize["Production"] / 1000000,
    marker="o",
    markersize=3,
    label="Actual Production"
)


# ------------------------------------------------------------
# TEST PREDICTIONS
# ------------------------------------------------------------

plt.plot(
    test["Year"],
    linear_prediction / 1000000,
    linestyle="--",
    label="Linear Test Prediction"
)

plt.plot(
    test["Year"],
    poly_prediction / 1000000,
    linestyle="--",
    label="Polynomial Test Prediction"
)

plt.plot(
    test["Year"],
    multi_prediction / 1000000,
    linestyle="--",
    label="Multivariate Test Prediction"
)


# ------------------------------------------------------------
# FUTURE PREDICTIONS
# ------------------------------------------------------------

plt.plot(
    future_results["Year"],
    future_results["Linear Prediction"] / 1000000,
    marker="o",
    linestyle=":",
    label="Linear Future Prediction"
)

plt.plot(
    future_results["Year"],
    future_results["Polynomial Prediction"] / 1000000,
    marker="s",
    linestyle=":",
    label="Polynomial Future Prediction"
)

plt.plot(
    future_results["Year"],
    future_results["Multivariate Prediction"] / 1000000,
    marker="^",
    linestyle=":",
    label="Multivariate Future Prediction"
)


# ------------------------------------------------------------
# TRAIN / TEST BOUNDARY
# ------------------------------------------------------------

plt.axvline(
    x=2011.5,
    linestyle="--",
    label="Train/Test Boundary"
)


# ------------------------------------------------------------
# GRAPH LABELS
# ------------------------------------------------------------

plt.title(
    "Maize Production: Actual, Test Predictions and Future Forecast"
)

plt.xlabel("Year")

plt.ylabel(
    "Maize Production (Million Tonnes)"
)

plt.legend(
    loc="upper left"
)

plt.grid(True)

plt.tight_layout()

plt.show()