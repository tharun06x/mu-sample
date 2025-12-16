"""
Real Estate Price Prediction using Linear Regression
----------------------------------------------------
This script trains a linear regression model to predict
house prices and saves the predictions to a CSV file.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_data(file_path):
    """Load dataset from CSV file"""
    return pd.read_csv(file_path)


def prepare_data(df, target_column):
    """Split features and target"""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def train_model(X_train, y_train):
    """Train Linear Regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_true, y_pred):
    """Evaluate model performance"""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2


def save_predictions(y_true, y_pred, output_path):
    """Save actual and predicted values to CSV"""
    results = pd.DataFrame({
        "Actual Price": y_true,
        "Predicted Price": y_pred
    })
    results.to_csv(output_path, index=False)


def plot_results(y_true, y_pred):
    """Generate evaluation plots"""
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted House Prices")
    plt.show()

    plt.figure()
    plt.scatter(y_pred, y_true - y_pred)
    plt.axhline(0)
    plt.xlabel("Predicted Prices")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()


def main():
    DATA_PATH = "Real_estate (1).csv"
    OUTPUT_PATH = "predictions_output.csv"
    TARGET_COLUMN = "Price"

    df = load_data(DATA_PATH)
    print("Dataset Summary:")
    print(df.describe())

    X, y = prepare_data(df, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    mse, r2 = evaluate_model(y_test, y_pred)
    print(f"\nMean Squared Error (MSE): {mse}")
    print(f"R² Score: {r2}")

    save_predictions(y_test, y_pred, OUTPUT_PATH)
    print(f"\nPredictions saved to {OUTPUT_PATH}")

    plot_results(y_test, y_pred)


if __name__ == "__main__":
    main()
