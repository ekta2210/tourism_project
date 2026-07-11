
from huggingface_hub import HfApi
import os

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload the deployment folder to Hugging Face Space
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id="<your_huggingface_username>/visit-with-us-prediction",
    repo_type="space",
    path_in_repo=""
)

print("Deployment files uploaded successfully to Hugging Face Space!")
