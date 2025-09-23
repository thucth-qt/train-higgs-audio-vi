from huggingface_hub import snapshot_download

# Define the repositories and their corresponding local directories
repos = {
    "bosonai/higgs-audio-v2-generation-3B-base": "/root/data/higgs/weights/higgs-audio-v2-generation-3B-base",
    "bosonai/higgs-audio-v2-tokenizer": "/root/data/higgs/weights/higgs-audio-v2-tokenizer"
}

# Download each repository to its specified local directory
for repo_id, local_dir in repos.items():
    print(f"Downloading {repo_id} to {local_dir} ...")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print(f"Finished downloading {repo_id}.")
