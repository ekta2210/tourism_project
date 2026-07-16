from huggingface_hub import HfApi
from huggingface_hub import create_repo
import os

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

create_repo(
    repo_id="ektasoni2210/visit-with-us-prediction",
    repo_type="space",
    space_sdk="docker",  # or "streamlit"/"gradio" depending on what's inside your `deployment` folder
    exist_ok=True,
    private=False,
    token=os.getenv("HF_TOKEN")
)

# Upload the deployment folder to Hugging Face Space
api.upload_folder(
    folder_path="deployment",
    repo_id="ektasoni2210/visit-with-us-prediction",
    repo_type="space",
    path_in_repo=""
)

print("Deployment files uploaded successfully to Hugging Face Space!")
