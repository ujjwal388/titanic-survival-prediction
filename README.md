# Titanic Survival Prediction

A beginner-friendly machine learning project that predicts whether a Titanic passenger survived using the Kaggle Titanic training dataset.

## 1. Objective

Build a binary classification model to predict the `Survived` target:
- `0` = Did not survive
- `1` = Survived

## 2. Dataset

Download `train.csv` from the [Kaggle Titanic competition data page](https://www.kaggle.com/competitions/titanic/data).

Place `train.csv` in the same folder as `titanic_prediction.py`.

The training file contains the target column `Survived`. The Kaggle `test.csv` file does not contain labels, so this project uses `train.csv` and creates its own held-out test split for evaluation.

## 3. Model selection

This project uses a **Random Forest Classifier**. It is a strong baseline for structured/tabular classification, can model nonlinear relationships, and provides feature-importance scores. It is not guaranteed to be the best possible model; performance should be compared with alternatives such as Logistic Regression, Decision Tree, and Gradient Boosting if time permits.

## 4. Data cleaning and preprocessing

The script:
- Prints dataset shape, data types, missing-value counts, and duplicate count.
- Removes exact duplicate rows.
- Creates `Title`, `FamilySize`, and `Deck` features.
- Drops identifier/raw text columns that are not directly used (`PassengerId`, `Name`, `Ticket`, `Cabin`).
- Imputes missing numeric values with the median.
- Imputes missing categorical values with the most frequent category.
- One-hot encodes categorical variables.
- Prints numeric correlations and saves a correlation heatmap.

Preprocessing is implemented inside a scikit-learn pipeline. This ensures imputation and encoding are fitted on the training data only, reducing data leakage.

## 5. Train/test split

The dataset is split into:
- 80% training data
- 20% testing data

The split is stratified by `Survived` so the class proportions are approximately preserved.

## 6. Evaluation

The program reports:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Classification report
- Confusion matrix

It also saves a feature-importance table and chart.

**Important:** Do not write a made-up accuracy in your report. Run the program and copy the actual metrics printed by your run. Results can vary with dataset version and model settings.

## 7. Installation and execution

Use Python 3.9 or newer.

```bash
pip install -r requirements.txt
python titanic_prediction.py --data train.csv
```

## 8. Output files

After running, the project creates:
- `correlation_heatmap.png`
- `survival_distribution.png`
- `confusion_matrix.png`
- `feature_importance.png`
- `feature_importance.csv`
- `titanic_survival_model.joblib`

## 9. Suggested project conclusion

The project demonstrates a complete supervised machine learning workflow, from dataset inspection and preprocessing to model training and evaluation. The Random Forest model learns patterns between passenger characteristics and survival outcomes. The evaluation metrics provide a more complete view of performance than accuracy alone. Feature importance can help identify which inputs contributed most to the model's predictions; however, feature importance does not establish causation.





