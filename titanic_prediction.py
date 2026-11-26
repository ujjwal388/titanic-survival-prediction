

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_score, recall_score, f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib


def add_features(df):
    """Create useful features while retaining the original input columns."""
    df = df.copy()

    if "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.", expand=False)
        df["Title"] = df["Title"].replace({
            "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"
        })
        df["Title"] = df["Title"].where(
            df["Title"].isin(["Mr", "Miss", "Mrs", "Master"]), "Rare"
        )

    if {"SibSp", "Parch"}.issubset(df.columns):
        df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1

    if "Cabin" in df.columns:
        df["Deck"] = df["Cabin"].astype("string").str[0]
        df["Deck"] = df["Deck"].fillna("Unknown")

    return df


def main():
    parser = argparse.ArgumentParser(description="Titanic survival prediction")
    parser.add_argument("--data", default="train.csv", help="Path to Kaggle train.csv")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. Download train.csv from the Kaggle Titanic competition."
        )

    df = pd.read_csv(data_path)
    print("\n--- Dataset overview ---")
    print("Shape:", df.shape)
    print(df.head())
    print("\nData types and non-null counts:")
    print(df.info())
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    # Remove exact duplicate records.
    df = df.drop_duplicates().copy()

    if "Survived" not in df.columns:
        raise ValueError("The training dataset must contain the 'Survived' target column.")

    # Target and features.
    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived"])

    # Feature engineering.
    X = add_features(X)

    # Exclude identifiers and high-cardinality/raw text fields.
    drop_columns = ["PassengerId", "Name", "Ticket", "Cabin"]
    X = X.drop(columns=[c for c in drop_columns if c in X.columns])

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    print("\nNumeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # Correlation analysis: numeric features plus target.
    correlation_data = X[numeric_features].copy()
    correlation_data["Survived"] = y
    corr = correlation_data.corr(numeric_only=True)
    print("\n--- Correlation with Survived ---")
    print(corr["Survived"].sort_values(ascending=False))

    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0)
    plt.title("Titanic Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.close()

    # Target distribution.
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title("Survival Count (0 = Did not survive, 1 = Survived)")
    plt.tight_layout()
    plt.savefig("survival_distribution.png", dpi=150)
    plt.close()

    # Preprocessing: median for numeric missing values; most frequent for categorical.
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Model evaluation ---")
    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, probabilities):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions,
        display_labels=["Did not survive", "Survived"],
        cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance using the fitted preprocessing feature names.
    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = pipeline.named_steps["classifier"].feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    print("\n--- Top 15 feature importances ---")
    print(importance_df.head(15).to_string(index=False))
    importance_df.to_csv("feature_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=importance_df.head(15),
        x="Importance", y="Feature"
    )
    plt.title("Top 15 Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

    joblib.dump(pipeline, "titanic_survival_model.joblib")
    print("\nSaved model: titanic_survival_model.joblib")
    print("Saved plots: correlation_heatmap.png, survival_distribution.png,")
    print("             confusion_matrix.png, feature_importance.png")


if __name__ == "__main__":
    main()
