# -----------------------------
# Import Libraries
# -----------------------------
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# -----------------------------
# MLflow Configuration
# -----------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("visit-with-us-training")

# -----------------------------
# Hugging Face API
# -----------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))

# -----------------------------
# Load Train/Test Data
# -----------------------------
Xtrain_path = "hf://datasets/ektasoni2210/visit-with-us-data/Xtrain.csv"
Xtest_path = "hf://datasets/ektasoni2210/visit-with-us-data/Xtest.csv"
ytrain_path = "hf://datasets/ektasoni2210/visit-with-us-data/ytrain.csv"
ytest_path = "hf://datasets/ektasoni2210/visit-with-us-data/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)

ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

# -----------------------------
# Feature Lists
# -----------------------------
numeric_features = [
    "Age",
    "CityTier",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch"
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched"
]

# -----------------------------
# Preprocessing
# -----------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# -----------------------------
# Model
# -----------------------------
xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# Hyperparameter Grid
# -----------------------------
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.01, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.8, 1.0]
}

# -----------------------------
# Pipeline
# -----------------------------
model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)

# -----------------------------
# Training
# -----------------------------
with mlflow.start_run():

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations
    results = grid_search.cv_results_

    for i in range(len(results["params"])):

        with mlflow.start_run(nested=True):

            mlflow.log_params(results["params"][i])
            mlflow.log_metric(
                "mean_cv_accuracy",
                results["mean_test_score"][i]
            )

    best_model = grid_search.best_estimator_

    mlflow.log_params(grid_search.best_params_)

    # -----------------------------
    # Predictions
    # -----------------------------
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    y_prob_train = best_model.predict_proba(Xtrain)[:, 1]
    y_prob_test = best_model.predict_proba(Xtest)[:, 1]

    # -----------------------------
    # Metrics
    # -----------------------------
    metrics = {
        "train_accuracy": accuracy_score(ytrain, y_pred_train),
        "test_accuracy": accuracy_score(ytest, y_pred_test),

        "train_precision": precision_score(ytrain, y_pred_train),
        "test_precision": precision_score(ytest, y_pred_test),

        "train_recall": recall_score(ytrain, y_pred_train),
        "test_recall": recall_score(ytest, y_pred_test),

        "train_f1": f1_score(ytrain, y_pred_train),
        "test_f1": f1_score(ytest, y_pred_test),

        "train_auc": roc_auc_score(ytrain, y_prob_train),
        "test_auc": roc_auc_score(ytest, y_prob_test)
    }

    mlflow.log_metrics(metrics)

    # -----------------------------
    # Save Model
    # -----------------------------
    model_path = "visit_with_us_model.joblib"

    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"Model saved as {model_path}")

    # -----------------------------
    # Upload Model to Hugging Face
    # -----------------------------
    repo_id = "ektasoni2210/visit-with-us-model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repository '{repo_id}' already exists.")

    except RepositoryNotFoundError:

        print("Creating model repository...")

        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=False
        )

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )

    print("Model uploaded successfully to Hugging Face.")
