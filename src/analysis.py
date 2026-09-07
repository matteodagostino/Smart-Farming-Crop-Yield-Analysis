"""Smart Farming Crop Yield Analysis

Portfolio version of the analysis completed for STA 4724 (Big Data Analytics Methods)
at the University of Central Florida.

This script focuses on Matteo D'Agostino's contribution:
- exploratory data analysis
- feature engineering
- simulated crop-yield construction
- decision-tree regression
- low-fertilizer performance analysis

Important: cropYield is an engineered/simulated target, not an observed agronomic yield.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split


DATA_PATH = "data/Crop_recommendationV2.csv"


def load_data(path=DATA_PATH):
    """Load the Smart Farming dataset."""
    return pd.read_csv(path)


def run_eda(df):
    """Run basic dataset quality and range checks."""
    print("Shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    for col in ["ph", "temperature", "rainfall"]:
        if col in df.columns:
            print(f"{col} range: {df[col].min():.3f} - {df[col].max():.3f}")


def engineer_features(df):
    """Create the derived variables used throughout the project."""
    data = df.copy()

    # Temperature-Humidity Index
    data["temperatureInFahrenheit"] = data["temperature"] * 9 / 5 + 32
    data["THI"] = 0.5 * (
        data["temperatureInFahrenheit"]
        + 61.0
        + ((data["temperatureInFahrenheit"] - 68.0) * 1.2)
        + (data["humidity"] * 0.094)
    )

    # Nutrient Balance Ratio
    data["NBR"] = (data["N"] + data["P"] + data["K"]) / 3

    # Water Availability Index
    data["WAI"] = data["soil_moisture"] + data["rainfall"]

    # Photosynthesis Potential
    data["PP"] = (
        data["sunlight_exposure"]
        * data["co2_concentration"]
        * data["temperature"]
    )

    # Soil Fertility Index
    data["SFI"] = data["organic_matter"] * (
        data["N"] + data["P"] + data["K"]
    )

    # Simulated yield variables
    data["estimatedAverageYieldPerPlant"] = (
        (
            data["fertilizer_usage"]
            + data["SFI"]
            + data["PP"]
        )
        * data["WAI"]
        * (data["growth_stage"] / 3)
        / (1 + data["pest_pressure"] + data["frost_risk"])
    )

    data["cropYield"] = (
        data["crop_density"] * data["estimatedAverageYieldPerPlant"]
    )

    data["logCropYield"] = np.log1p(data["cropYield"])

    return data


def plot_yield_distributions(data):
    """Compare simulated yield before and after log transformation."""
    plt.figure(figsize=(8, 5))
    plt.hist(data["cropYield"], bins=30)
    plt.title("Distribution of Estimated Crop Yield")
    plt.xlabel("Estimated Crop Yield")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(data["logCropYield"], bins=30)
    plt.title("Distribution of Log Estimated Crop Yield")
    plt.xlabel("Log Estimated Crop Yield")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def train_decision_tree(data):
    """Tune and evaluate the Decision Tree Regressor."""
    features = [
        "fertilizer_usage",
        "WAI",
        "rainfall",
        "PP",
        "THI",
        "pest_pressure",
        "frost_risk",
        "growth_stage",
        "crop_density",
    ]

    X = data[features]
    y = data["logCropYield"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=1,
    )

    parameter_grid = {
        "max_depth": np.arange(2, 11),
        "min_samples_leaf": np.arange(1, 31, 5),
    }

    search = GridSearchCV(
        estimator=tree.DecisionTreeRegressor(random_state=1),
        param_grid=parameter_grid,
        cv=KFold(n_splits=5, shuffle=True, random_state=1),
    )
    search.fit(X_train, y_train)

    model = tree.DecisionTreeRegressor(
        max_depth=search.best_params_["max_depth"],
        min_samples_leaf=search.best_params_["min_samples_leaf"],
        random_state=1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nBest parameters:", search.best_params_)
    print("Test MSE:", mean_squared_error(y_test, predictions))
    print("Test R²:", r2_score(y_test, predictions))

    return model, features


def analyze_low_fertilizer(data, model, features):
    """Compare predicted performance under low fertilizer usage."""
    threshold = data["fertilizer_usage"].quantile(0.25)

    low_fertilizer = data[
        data["fertilizer_usage"] <= threshold
    ].copy()

    low_fertilizer["predicted_logYield"] = model.predict(
        low_fertilizer[features]
    )

    low_log_mean = low_fertilizer["predicted_logYield"].mean()
    overall_log_mean = data["logCropYield"].mean()

    low_yield = np.exp(low_log_mean)
    overall_yield = np.exp(overall_log_mean)
    percent_difference = (
        (overall_yield - low_yield) / overall_yield * 100
    )

    print("\nLow-fertilizer threshold:", threshold)
    print("Average predicted log yield under low fertilizer:", low_log_mean)
    print("Overall average log crop yield:", overall_log_mean)
    print("Estimated low-fertilizer yield:", low_yield)
    print("Estimated overall yield:", overall_yield)
    print("Estimated reduction (%):", percent_difference)

    return low_fertilizer


def compare_low_fertilizer_conditions(data, low_fertilizer, features):
    """Compare high- and low-performing observations under low fertilizer."""
    median_yield = data["logCropYield"].median()

    high_group = low_fertilizer[
        low_fertilizer["predicted_logYield"] >= median_yield
    ]
    low_group = low_fertilizer[
        low_fertilizer["predicted_logYield"] < median_yield
    ]

    variables = features + ["SFI"]

    comparison = pd.DataFrame(
        {
            "High Yield + Low Fertilizer": high_group[variables].mean(),
            "Low Yield + Low Fertilizer": low_group[variables].mean(),
        }
    )

    print("\nCondition comparison:")
    print(comparison)

    return comparison


def main():
    df = load_data()
    run_eda(df)

    data = engineer_features(df)
    plot_yield_distributions(data)

    model, features = train_decision_tree(data)
    low_fertilizer = analyze_low_fertilizer(data, model, features)
    compare_low_fertilizer_conditions(data, low_fertilizer, features)


if __name__ == "__main__":
    main()
