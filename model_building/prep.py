# -----------------------------
# Import Required Libraries
# -----------------------------
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from huggingface_hub import HfApi

# -----------------------------
# Hugging Face Configuration
# -----------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = "hf://datasets/ektasoni2210/visit-with-us-data/tourism.csv"

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print(df.head())

# -----------------------------
# Data Cleaning
# -----------------------------

# Drop unnecessary column
df.drop(columns=["CustomerID"], inplace=True)

# Handle missing values

categorical_cols = df.select_dtypes(include="object").columns
numerical_cols = df.select_dtypes(exclude="object").columns

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# -----------------------------
# Encode Categorical Variables
# -----------------------------
label_encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# -----------------------------
# Define Features and Target
# -----------------------------
target_col = "ProdTaken"

X = df.drop(columns=[target_col])
y = df[target_col]

# -----------------------------
# Train-Test Split
# -----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------
# Save Files
# -----------------------------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train and Test datasets saved successfully.")

# -----------------------------
# Upload Files to Hugging Face
# -----------------------------
files = [
    "Xtrain.csv",
    "Xtest.csv",
    "ytrain.csv",
    "ytest.csv"
]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="<your_huggingface_username>/visit-with-us-data",
        repo_type="dataset",
    )

print("Files uploaded successfully to Hugging Face.")
