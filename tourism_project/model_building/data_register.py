from huggingface_hub import HfApi, create_repo
import os

# -----------------------------
# Hugging Face Dataset Repository
# -----------------------------
repo_id = "ektasoni2210/tourism-package-dataset"
repo_type = "dataset"

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Create the dataset repository if it doesn't exist
create_repo(
    repo_id=repo_id,
    repo_type=repo_type,
    private=False,
    exist_ok=True
)

print(f"Dataset repository '{repo_id}' is ready.")

# Upload the complete data folder
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Data uploaded successfully to Hugging Face Dataset Hub!")
