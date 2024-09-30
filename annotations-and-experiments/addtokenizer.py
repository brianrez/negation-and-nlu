import os
import shutil
from argparse import ArgumentParser

from huggingface_hub import HfApi, Repository, login
from transformers import AutoTokenizer

args = ArgumentParser()
args.add_argument('--model_name', type=str, default='roberta-base')
args.add_argument('--model_path', type=str, default='roberta-base')
args = args.parse_args()

# Login to the Hugging Face Hub
# Replace 'your_huggingface_token' with your actual token
# login(token='your_huggingface_token')

# Specify the repository details
repo_id = args.model_path
repo_local_dir = "./" + args.model_path

# Clone the repository to a local directory
repo = Repository(local_dir=repo_local_dir, clone_from=repo_id)

# Download the tokenizer files from the `roberta-base` model
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer_dir = os.path.join(repo_local_dir, args.model_name + "-tokenizer")
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer.save_pretrained(tokenizer_dir)

# List of tokenizer files
tokenizer_files = ["vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"]

# Copy tokenizer files to the custom model directory
for file_name in tokenizer_files:
    src_file = os.path.join(tokenizer_dir, file_name)
    dest_file = os.path.join(repo_local_dir, file_name)
    shutil.copyfile(src_file, dest_file)

# Add and commit the files to the repository
repo.git_add(auto_lfs_track=True)
repo.git_commit("Add roberta-base tokenizer files")

# Push the changes to the Hugging Face Hub
repo.git_push()

print("Tokenizer files have been added and pushed to the Hugging Face Hub.")
