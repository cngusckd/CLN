#!/bin/bash

# Use Python to generate all combinations of configuration options
combinations=$(python - <<END
import itertools

# Define the possible values for each configuration option
datasets = ["mnist", "cifar10", "cifar100"]  # List of datasets to be used
cl_types = ["cil", "dil"]  # Types of Continual Learning approaches
models = ["er", "er_ace", "der", "der++"]  # Different model types to test
buffer_extractions = ["random", "mir"]  # Methods for buffer extraction
buffer_storages = ["random", "gss"]  # Methods for buffer storage

# Generate all possible combinations of the configuration options
for combination in itertools.product(datasets, cl_types, models, buffer_extractions, buffer_storages):
    # Print each combination as a space-separated string
    print(" ".join(combination))
END
)

# Iterate over each generated combination
while IFS= read -r combo; do
  # Split the combination string into individual variables
  set -- $combo
  dataset=$1  # Assign the first element to the dataset variable
  cl_type=$2  # Assign the second element to the cl_type variable
  model=$3  # Assign the third element to the model variable
  buffer_extraction=$4  # Assign the fourth element to the buffer_extraction variable
  buffer_storage=$5  # Assign the fifth element to the buffer_storage variable

  # Display the current configuration being tested
  echo "Running with dataset=$dataset, cl_type=$cl_type, model=$model, buffer_extraction=$buffer_extraction, buffer_storage=$buffer_storage"

  # Execute the main.py script with the current configuration
  # The script is run with the specified dataset, cl_type, model, buffer_extraction, and buffer_storage
  # The --wandb flag is used to enable logging with Weights & Biases
  # The --epoch 2 flag sets the number of epochs to 2 for this test
  python main.py --wandb --dataset $dataset --epoch 2 --cl_type $cl_type --model $model --buffer_extraction $buffer_extraction --buffer_storage $buffer_storage
done <<< "$combinations"