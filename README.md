# Dataset Tools
This repository holds a collection of scripts to assist in dataset generation, however it is currently a work in progress! Many of these scripts won't work out of the box on Windows, as the forward slash `/` has been used as the file separator (will update ASAP).

Additionally, some scripts have minor bugs, those will be fixed ASAP as well. 

This repo is intended to gather all the 'dataset' scripts together and make sure they are up to date, so feel free to add to it/alter it, fix bugs, etc.

Currently there are starting points for augmenting images, testing datasets (overlaying bounding boxes on images to sanity check), updating labels, and updating file endings (e.g. 'png' to 'jpg').

Please put any `TODO` items you think would be good on the TODO list below!

Anytime you work on a new item, please create a new branch. Once you're done with your work, make a PR and try to add at least one person from a team (e.g. web, edgeIQ, etc.)

## TODO:
- clean up scripts (this is a reminder for Lila: OS agnostic file reading, input parameters)
- Future work for `model-evaluation`: enable other subsets for F1 and accuracy calculation (e.g. box size or quadrant)

## Augmentation
You can run as a script (see below), for which you'll need to run `pip3 install imgaug` to install the necessary augmentation library. Otherwise, you can create an alwaysAI project, modify the `alwaysai.app.json` to run `augment_images.py` and create a `requirements.txt` that includes `imgaug`, and use the typical alwaysAI workflow. We will probably re-structure the project to use the latter workflow, this is just an option for now.

The script `augment_images.py` and the notebook `augment_images.ipynb` enable augmentation of a Pascal VOC, zipped dataset.

To run as a Jupyter Notebook:
Run `jupyter notebook` in your terminal from within the local directory containing the notebook `augment_images.ipynb`. You should be taken to your Jupyter Notebook home, and you should see all the local files in this home directory. Make sure the input dataset you want to work with is in this directory. Then open the notebook. Specify your input and output directories (in .zip format) and run all the initial cells, until you see the comment for choosing specific augmentations. You can run any of the augmentation cells independently after this point. Finally, run the last cell to zip the new augmented dataset together.

To run as a script:
```
$ python3 augment_images.py --input_dir <input.zip> --output_dir <output.zip>
```

## Annotation Verification (Box Overlay)
The script `test_annotations.py` and the Jupyter Notebook `test_annotations.ipynb` enable you to overlay annotations on images, which is useful when you have an open
source dataset, or have performed manipulations on annotations and want to quality check. Note, you may also upload datasets to CVAT, however this enables you to perform this locally and specify a sample rate.

To run the `test_annotations.ipynb` notebook:
Run `jupyter notebook` in your terminal, and then open the `test_annotations.ipynb`. You should be taken to your Jupyter Notebook home, and you should see all the local files in this home directory. Make sure the input dataset you want to work with is in this directory. Specify the input and output dataset in the appropriate cells, as well as the `sample` rate, and then run all the subsequent cells.

Note: if you don't provide an argument for `--sample`, the code will output overy test pair; otherwise, specify an integer of the sample value. The sample value means the code will output every `sample` pairs (e.g. if there are 4 images and `sample` is 2, then there will be 4 / 2 = 2 images output. If sample is not provided, or is 1, then all images are output, e.g. 4 / 1 = 4). If the sample rate is greater than the number of pairs, the number of pairs will be used as the sample value.

To run the `test_ann.py` script:
```
$ python3 test_ann.py --input_dir <input.zip> --output_dir <output.zip>
```
Note: if you don't provide an argument for `--sample`, the code will output overy test pair; otherwise, specify an integer of the sample value. The sample value means the code will output every `sample` pairs (e.g. if there are 4 images and `sample` is 2, then there will be 4 / 2 = 2 images output. If sample is not provided, or is 1, then all images are output, e.g. 4 / 1 = 4). If the sample rate is greater than the number of pairs, the number of pairs will be used as the sample value.

## Dataset Analysis
This repo included some helpful tools for reading in JSON detections, plotting object detection boxes, etc. This repo was used in to assess training vs testing datasets, to determine any gaps in the training footage, for example.

## Auto Annotation
This application (found in `auto-annotate`) uses an initial model to generate annotations, which can be read into CVAT and edited.

## Model Evaluation
This application (found in `model-evaluation`) enables you to assess model performance based on overall F1 and accuracy, as well as a per-label F1 and accuracy. Future improvements include calculating F1 and accuracy by other factors, such as bounding box size, bounding box location in the frame, etc.

## JSON to Pascal VOC
Found in `jsons-to-pascal-voc`. This application converts JSON datasets to Pascal VOC for compatibility with the alwaysAI system.

## Class Balancer
This application, available with `class_balancer.py`, will calculate the minimum dataset that contains balanced classes (labels).