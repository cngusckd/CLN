#!/bin/bash

# Use Python to generate all combinations of configuration options
combinations=$(python - <<END
import itertools

# Define the possible values for each configuration option
datasets = ["mnist", "cifar10", "cifar100"]
cl_types = ["cil", "dil"]
models = ["er", "er_ace", "der", "der++"]
buffer_extractions = ["random", "mir"]
buffer_storages = ["random", "gss"]

# Generate all possible combinations of the configuration options
for combination in itertools.product(datasets, cl_types, models, buffer_extractions, buffer_storages):
    # Print each combination as a space-separated string
    print(" ".join(combination))
END
)

# Function to run a single combination
run_combination() {
  combo=$1
  # Split the combination string into individual variables
  set -- $combo
  dataset=$1
  cl_type=$2
  model=$3
  buffer_extraction=$4
  buffer_storage=$5

  # Display the current configuration being tested
  echo "Running with dataset=$dataset, cl_type=$cl_type, model=$model, buffer_extraction=$buffer_extraction, buffer_storage=$buffer_storage"

  # Execute the main.py script with the current configuration
  python main.py --wandb --dataset $dataset --epoch 2 --cl_type $cl_type --model $model --buffer_extraction $buffer_extraction --buffer_storage $buffer_storage
}

# Export the function so it can be used by xargs
export -f run_combination

# Iterate over each generated combination and run them in parallel
echo "$combinations" | xargs -n 1 -P 5 -I {} bash -c 'run_combination "$@"' _ {}