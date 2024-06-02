#!/bin/bash

# Define your inputs here. For example:
inputs=(
    "--task_name qnli --model_name roberta-base --model_path mhr2004/roberta-base-nsp-1000000-1e-06-32 --batch_size 16 --learning_rate 1e-6"
    "--task_name qnli --model_name roberta-base --model_path mhr2004/roberta-base-nsp-1000000-1e-06-32 --batch_size 16 --learning_rate 5e-6"
    "--task_name qnli --model_name roberta-base --model_path mhr2004/roberta-base-nsp-1000000-1e-06-32 --batch_size 16 --learning_rate 1e-5"
    "--task_name qnli --model_name roberta-base --model_path mhr2004/roberta-base-nsp-1000000-1e-06-32 --batch_size 16 --learning_rate 5e-5"
    "--task_name qnli --model_name roberta-base --model_path mhr2004/roberta-base-nsp-1000000-1e-06-32 --batch_size 16 --learning_rate 1e-4"
        
    "--task_name qnli --model_name roberta-base --model_path mhr2004/roberta-base-pp-1000000-1e-06-128 --batch_size 16 --learning_rate 1e-6"
    "--task_name qnli --model_name roberta-base --model_path mhr2004/roberta-base-pp-1000000-1e-06-128 --batch_size 16 --learning_rate 5e-6"
    "--task_name qnli --model_name roberta-base --model_path mhr2004/roberta-base-pp-1000000-1e-06-128 --batch_size 16 --learning_rate 1e-5"
    "--task_name qnli --model_name roberta-base --model_path mhr2004/roberta-base-pp-1000000-1e-06-128 --batch_size 16 --learning_rate 5e-5"
    "--task_name qnli --model_name roberta-base --model_path mhr2004/roberta-base-pp-1000000-1e-06-128 --batch_size 16 --learning_rate 1e-4"

    "--task_name qnli --model_name roberta-base --model_path mhr2004/roberta-base-dual-1000000-1e-06-128 --batch_size 16 --learning_rate 1e-6"
    "--task_name qnli --model_name roberta-base --model_path mhr2004/roberta-base-dual-1000000-1e-06-128 --batch_size 16 --learning_rate 5e-6"
    "--task_name qnli --model_name roberta-base --model_path mhr2004/roberta-base-dual-1000000-1e-06-128 --batch_size 16 --learning_rate 1e-5"
    "--task_name qnli --model_name roberta-base --model_path mhr2004/roberta-base-dual-1000000-1e-06-128 --batch_size 16 --learning_rate 5e-5"
    "--task_name qnli --model_name roberta-base --model_path mhr2004/roberta-base-dual-1000000-1e-06-128 --batch_size 16 --learning_rate 1e-4"


    "--task_name wsc --model_name roberta-base --model_path mhr2004/roberta-base-nsp-1000000-1e-06-32 --batch_size 16 --learning_rate 1e-6"
    "--task_name wsc --model_name roberta-base --model_path mhr2004/roberta-base-nsp-1000000-1e-06-32 --batch_size 16 --learning_rate 5e-6"
    "--task_name wsc --model_name roberta-base --model_path mhr2004/roberta-base-nsp-1000000-1e-06-32 --batch_size 16 --learning_rate 1e-5"
    "--task_name wsc --model_name roberta-base --model_path mhr2004/roberta-base-nsp-1000000-1e-06-32 --batch_size 16 --learning_rate 5e-5"
    "--task_name wsc --model_name roberta-base --model_path mhr2004/roberta-base-nsp-1000000-1e-06-32 --batch_size 16 --learning_rate 1e-4"
        
    "--task_name wsc --model_name roberta-base --model_path mhr2004/roberta-base-pp-1000000-1e-06-128 --batch_size 16 --learning_rate 1e-6"
    "--task_name wsc --model_name roberta-base --model_path mhr2004/roberta-base-pp-1000000-1e-06-128 --batch_size 16 --learning_rate 5e-6"
    "--task_name wsc --model_name roberta-base --model_path mhr2004/roberta-base-pp-1000000-1e-06-128 --batch_size 16 --learning_rate 1e-5"
    "--task_name wsc --model_name roberta-base --model_path mhr2004/roberta-base-pp-1000000-1e-06-128 --batch_size 16 --learning_rate 5e-5"
    "--task_name wsc --model_name roberta-base --model_path mhr2004/roberta-base-pp-1000000-1e-06-128 --batch_size 16 --learning_rate 1e-4"

    "--task_name wsc --model_name roberta-base --model_path mhr2004/roberta-base-dual-1000000-1e-06-128 --batch_size 16 --learning_rate 1e-6"
    "--task_name wsc --model_name roberta-base --model_path mhr2004/roberta-base-dual-1000000-1e-06-128 --batch_size 16 --learning_rate 5e-6"
    "--task_name wsc --model_name roberta-base --model_path mhr2004/roberta-base-dual-1000000-1e-06-128 --batch_size 16 --learning_rate 1e-5"
    "--task_name wsc --model_name roberta-base --model_path mhr2004/roberta-base-dual-1000000-1e-06-128 --batch_size 16 --learning_rate 5e-5"
    "--task_name wsc --model_name roberta-base --model_path mhr2004/roberta-base-dual-1000000-1e-06-128 --batch_size 16 --learning_rate 1e-4"



    "--task_name wic --model_name roberta-base --model_path mhr2004/roberta-base-nsp-1000000-1e-06-32 --batch_size 16 --learning_rate 1e-6"
    "--task_name wic --model_name roberta-base --model_path mhr2004/roberta-base-nsp-1000000-1e-06-32 --batch_size 16 --learning_rate 5e-6"
    "--task_name wic --model_name roberta-base --model_path mhr2004/roberta-base-nsp-1000000-1e-06-32 --batch_size 16 --learning_rate 1e-5"
    "--task_name wic --model_name roberta-base --model_path mhr2004/roberta-base-nsp-1000000-1e-06-32 --batch_size 16 --learning_rate 5e-5"
    "--task_name wic --model_name roberta-base --model_path mhr2004/roberta-base-nsp-1000000-1e-06-32 --batch_size 16 --learning_rate 1e-4"
        
    "--task_name wic --model_name roberta-base --model_path mhr2004/roberta-base-pp-1000000-1e-06-128 --batch_size 16 --learning_rate 1e-6"
    "--task_name wic --model_name roberta-base --model_path mhr2004/roberta-base-pp-1000000-1e-06-128 --batch_size 16 --learning_rate 5e-6"
    "--task_name wic --model_name roberta-base --model_path mhr2004/roberta-base-pp-1000000-1e-06-128 --batch_size 16 --learning_rate 1e-5"
    "--task_name wic --model_name roberta-base --model_path mhr2004/roberta-base-pp-1000000-1e-06-128 --batch_size 16 --learning_rate 5e-5"
    "--task_name wic --model_name roberta-base --model_path mhr2004/roberta-base-pp-1000000-1e-06-128 --batch_size 16 --learning_rate 1e-4"

    "--task_name wic --model_name roberta-base --model_path mhr2004/roberta-base-dual-1000000-1e-06-128 --batch_size 16 --learning_rate 1e-6"
    "--task_name wic --model_name roberta-base --model_path mhr2004/roberta-base-dual-1000000-1e-06-128 --batch_size 16 --learning_rate 5e-6"
    "--task_name wic --model_name roberta-base --model_path mhr2004/roberta-base-dual-1000000-1e-06-128 --batch_size 16 --learning_rate 1e-5"
    "--task_name wic --model_name roberta-base --model_path mhr2004/roberta-base-dual-1000000-1e-06-128 --batch_size 16 --learning_rate 5e-5"
    "--task_name wic --model_name roberta-base --model_path mhr2004/roberta-base-dual-1000000-1e-06-128 --batch_size 16 --learning_rate 1e-4"
)

# Maximum number of concurrent jobs, equals to the number of GPUs
MAX_JOBS=4

# Function to check and wait for available job slot
check_jobs() {
  while [ $(jobs -p | wc -l) -ge $MAX_JOBS ]; do
    sleep 2
  done
}

# Loop through inputs and execute them
for i in "${!inputs[@]}"; do
  # Check if we need to wait for a job slot to become available
  check_jobs
  
  # Calculate GPU index: i % MAX_JOBS ensures cycling through GPUs 0 to MAX_JOBS-1
  gpu_index=$((i % MAX_JOBS))

  echo "Starting job $i on GPU $gpu_index"
  echo "CUDA_VISIBLE_DEVICES=$gpu_index python3 nlutrainer.py ${inputs[$i]}"

  # Execute the script with CUDA_VISIBLE_DEVICES set for the specific GPU
  CUDA_VISIBLE_DEVICES=$gpu_index python3 nlutrainer.py ${inputs[$i]} &
done

# Wait for all background jobs to finish
wait

echo "All processes completed"

curl -d "condaqa trainings finished" ntfy.sh/mhrnlpmodels 