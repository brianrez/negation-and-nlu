#!/bin/bash

input=(
  "--model_name bert-large-uncased --model_path bert-large-uncased-nsp-500000-1e-06-128"
  "--model_name bert-large-uncased --model_path bert-large-uncased-pp-500000-1e-06-128"
  "--model_name bert-large-uncased --model_path bert-large-uncased-dual-500000-1e-06-32"
  "--model_name bert-base-uncased --model_path bert-base-uncased-nsp-500000-1e-06-64"
  "--model_name bert-base-uncased --model_path bert-base-uncased-pp-500000-1e-06-64"
  "--model_name bert-base-uncased --model_path bert-base-uncased-dual-500000-1e-06-32"
)

for i in "${!input[@]}"; do
  echo "Starting job $i"
  echo "python3 addtokenizer.py ${input[$i]}"
  python3 addtokenizer.py ${input[$i]}
done