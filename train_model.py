"""
train_model.py
----------------
Trains a K-Nearest Neighbors classifier on the Iris dataset (from Module 4)
and saves the trained model + scaler to disk so the Streamlit app can load
them instantly without retraining every time.

Run this once before starting the app:
    python train_model.py
"""

import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def main():
    # 1. Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    # 5. Quick sanity check
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"Test accuracy: {accuracy:.2%}")

    # 6. Save model + scaler + class names for the app to use
    joblib.dump(model, "model/iris_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(list(iris.target_names), "model/target_names.pkl")

    print("Saved model, scaler, and target names to the 'model/' folder.")


if __name__ == "__main__":
    main()
