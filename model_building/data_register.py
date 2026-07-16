from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# -----------------------------
# Hugging Face Dataset Repository
# -----------------------------
repo_id = "ektasoni2210/visit-with-us-data"
repo_type = "dataset"

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if the dataset repository exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repository '{repo_id}' already exists. Using it.")

except RepositoryNotFoundError:
    print(f"Dataset repository '{repo_id}' not found. Creating a new repository...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False
    )
    print(f"Dataset repository '{repo_id}' created successfully.")

# Upload the complete data folder
api.upload_folder(
    folder_path="data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Data uploaded successfully to Hugging Face Dataset Hub!")
