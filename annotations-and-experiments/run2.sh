#!/bin/bash
# Define your inputs here. For example:
inputs=()

lrs=("1e-6" "5e-6" "1e-5" "5e-5" "1e-4" "2e-5")
batch_sizes=("16" "8")
tasks=("qnli" "wsc" "wic")
model_pathes=(
  "--model_name bert-large-uncased --model_path mhr2004/bert-large-uncased-nsp-500000-1e-06-128"
  "--model_name bert-large-uncased --model_path mhr2004/bert-large-uncased-nsp-1000000-1e-06-64"
  "--model_name bert-large-uncased --model_path mhr2004/bert-large-uncased-pp-500000-1e-06-128"
  "--model_name bert-large-uncased --model_path mhr2004/bert-large-uncased-pp-1000000-1e-06-32"
  "--model_name bert-large-uncased --model_path mhr2004/bert-large-uncased-dual-500000-1e-06-32"
  "--model_name bert-large-uncased --model_path mhr2004/bert-large-uncased-dual-1000000-1e-06-32"
  "--model_name bert-large-uncased --model_path bert-large-uncased"
  "--model_name bert-base-uncased  --model_path mhr2004/bert-base-uncased-nsp-500000-1e-06-64"
  "--model_name bert-base-uncased  --model_path mhr2004/bert-base-uncased-nsp-1000000-1e-06-32"
  "--model_name bert-base-uncased  --model_path mhr2004/bert-base-uncased-pp-500000-1e-06-64"
  "--model_name bert-base-uncased  --model_path mhr2004/bert-base-uncased-pp-1000000-1e-06-32"
  "--model_name bert-base-uncased  --model_path mhr2004/bert-base-uncased-dual-500000-1e-06-32"
  "--model_name bert-base-uncased  --model_path mhr2004/bert-base-uncased-dual-1000000-1e-06-32"
  "--model_name bert-base-uncased  --model_path bert-base-uncased"
)

for model_path in "${model_pathes[@]}"; do
  for lr in "${lrs[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
      for task in "${tasks[@]}"; do
        inputs+=("$model_path  --task_name $task --learning_rate $lr --batch_size $batch_size")
      done
    done
  done
done


# Maximum number of concurrent jobs, equals to the number of GPUs
MAX_JOBS=4

# Function to check and wait for available job slot
check_jobs() {
  while [ $(jobs -p | wc -l) -ge $MAX_JOBS ]; do
    sleep 10
  done
}

# Loop through inputs and execute them
for i in "${!inputs[@]}"; do
  # Check if we need to wait for a job slot to become available
  check_jobs

  # wait a minute before starting the next job
  sleep 60
  
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