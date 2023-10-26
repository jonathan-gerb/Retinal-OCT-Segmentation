#!/bin/bash

# go to project directory
cd "/home/parting/master_AI/medical_ai/Fundus-OCT-challenge"

# List of all the experiment folders
folders=(
    "experiment_SSL_vs_imagenet"
    "experiment_subset25_transforms"
    "experiment_subset5_transforms"
    "experiment_transforms"
    "experiment_SSL_imagenet_subset5"
    "experiment_SSL_imagenet_subset25"
)

fundus-oct --cfg run/configs/experiment_transforms/goals_finetuning_basicunet_alltransform.yml

# Loop through each folder
for folder in "${folders[@]}"; do
    # Find all the .yml files in the folder and run the command
    for config in run/configs/$folder/*.yml; do
        echo "Running: fundus-oct --cfg $config"
        fundus-oct --cfg $config
    done
done

echo "All experiments completed!"