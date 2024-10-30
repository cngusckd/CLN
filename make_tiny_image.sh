#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# URL for downloading the Tiny ImageNet dataset
URL="http://cs231n.stanford.edu/tiny-imagenet-200.zip"

# Directory to store the dataset
DATASET_DIR="data/tiny_imagenet"

# Create the dataset directory
mkdir -p $DATASET_DIR

# Download the dataset
echo "Downloading Tiny ImageNet dataset..."
wget $URL -O $DATASET_DIR/tiny-imagenet-200.zip

# Unzip the dataset
echo "Unzipping dataset..."
unzip $DATASET_DIR/tiny-imagenet-200.zip -d $DATASET_DIR

# Remove the zip file
rm $DATASET_DIR/tiny-imagenet-200.zip

# Organize the validation set into subdirectories
VAL_DIR="$DATASET_DIR/tiny-imagenet-200/val"
ANNOTATIONS_FILE="$VAL_DIR/val_annotations.txt"

echo "Organizing validation images into subdirectories..."
while read -r line; do
    # Read the image file name and the corresponding class label
    IMAGE_FILE=$(echo $line | awk '{print $1}')
    CLASS_LABEL=$(echo $line | awk '{print $2}')
    
    # Create the class directory if it doesn't exist
    CLASS_DIR="$VAL_DIR/$CLASS_LABEL"
    mkdir -p $CLASS_DIR
    
    # Move the image file to the class directory
    mv "$VAL_DIR/images/$IMAGE_FILE" "$CLASS_DIR/"
done < "$ANNOTATIONS_FILE"

# Remove the images directory as it's no longer needed
rm -r "$VAL_DIR/images"

# Check the directory structure
echo "Dataset structure:"
tree $DATASET_DIR/tiny-imagenet-200

echo "Tiny ImageNet dataset is ready."