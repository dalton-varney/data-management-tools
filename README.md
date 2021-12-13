# Dataset Tools
This repository holds a collection of scripts to assist in dataset generation.

Currently there are starting points for augmenting images, balancing datasets, generating synthetic datasets, and renaming files.

For all augmentation, class balancer, you can create a new alwaysAI project and specify which Python file to run in the `alwaysai.app.json`.

## Label Converter
This script can be used to rename certain labels in your dataset. This script requires the use of `label_mappings.csv` -- you can specify the bad labels in the first column, and the corresponding label to replace the bad label with in the second column. A dummy `label_mappings.csv` has been provided.

## Augmentation
The script `augment_images.py` enables augmentation of a Pascal VOC, zipped dataset. You can specify which combinations of augmentation you want to run using the input arguments.

## Class Balancer
This application, available with `class_balancer.py`, will calculate the minimum dataset that contains balanced classes (labels) and print the files that should be removed from the current dataset. You can specify the input directory (which is an unzipped dataset) on line 175.

## Synthetic Objects
This project generates synthetic data. It is expected that you have the `results.xml` file as well as a folder with named `Classes` that contains `.png` files. You also need to have a folder of `Annotations`, as well as `backgrounds`.