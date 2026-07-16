from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

SPACE_REPO = "ektasoni2210/superkart-frontend"  # reuse existing Space, no create_repo needed

api.upload_folder(
    folder_path="deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
)

print(f"Deployment files uploaded to {SPACE_REPO}")
