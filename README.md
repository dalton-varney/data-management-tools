# Dataset Tools
This repository holds a collection of scripts to assist in dataset generation and alteration.

Currently there are scripts for augmenting images, overlaying annotations on images, balancing datasets, generating synthetic datasets, and converting labels.

For the scripts in the upper directory (i.e. not the sythetic dataset project), you can create a new alwaysAI project and specify which Python file to run in the `alwaysai.app.json`. 

## Augmentation
The script `augment_images.py` enables augmentation of a Pascal VOC, zipped dataset. You can specify which combinations of augmentation you want to run using the input arguments. The following lines of code will use the `input_dir` as the input directory and it will perform two augmentations on the original images: 1). it will brighten them, and 2). it will rotate them 180 degrees and make them grayscale. After you change the `alwaysai.app.json` to specify `augment_iamges.py`,

run the code with  

```aai app start -- --input_dir <path/to/dir> --brighten --rotate_180_grayscale```

## Test Annotations
This script will overlay all the annotations over their corresponding image in a zipped Pascal VOC dataset, to test that the annotations are lining up properly. You must specify the input directory, using the `--input_dir` flag. You may specify an output directory, using the `--output_dir` flag, however if you do not specify this flag, an output filename will be generated for you. You can also use the `--sample` flag, along with an integer parameter, to specify how many images you want to test. If you use `--sample 10`, it will test 1 image in 10. If you don't use this flag, all images in the dataset will be tested.

You can run this by changing the Run section in your `alwaysai.app.json` to be `test_annotations.py` and using:

```aai app start -- --input_dir <path/to/dir> --output_dir <path/to/dir> --sample <int>```

## Class Balancer
This application, available with `class_balancer.py`, will calculate the minimum dataset that contains balanced classes (labels) and print the files that should be removed from the current dataset. If you don't use the `--partition` flag, you will just get a print out of how many files should be removed from your dataset. If you do use the `--partition` flag, it will create a sub-directory inside your input directory with the image/annotations pairs you should hold out. If you specify the `--output_dir` flag and a name, that name will be used for the new holdout folder, otherwise it will automatically use 'holdout'. For example, you can run this code by change the `alwaysai.app.json` to run the `class_balancer.py` file and using the following command: 

```aai app start -- --input_dir <path/to/dir> --output_dir <path/to/dir> --sample <int>```

## Synthetic Objects
This project generates synthetic data. It is expected that you have the `results.xml` file as well as a folder with named `Classes` that contains one or more sub-folders, which then contain `.png` files. files You also need to have a folder of `Annotations`, as well as a folder of `JPEGImages`, which are both empty directories. The `Annotations` and `JPEGImages` empty directories will be created if they do not exist. You will need to create a `backgrounds` folder, which should contain images. These can be either `.jpg` or `.png`, but be aware that `.png` files will result in a very large directory. You run this application as an alwaysAI project. 

The folder structure should look as follows:
```
    data-management-tools
        synthetic-objects
            synth.py
            Annotations
            JPEGImages
            Classes
                <some-folder>
                    *.png
            results.xml
```         

Change into the `synthetic-objects` folder, and run the `aai app configure`. An `app.py` file will be created, press `yes`. Modify the `alwaysai.app.json` file to run `class_balancer.py`. Then run `aai app install` and then `aai app start` to run the project.

## Label Converter
This script can be used to rename certain labels in your dataset. This script requires the use of `label_mappings.csv` -- you can specify the bad labels in the first column, and the corresponding label to replace the bad label with in the second column. There is no header row expected. The expected format is:
```
colored_plastic_bottle,bottle
clear_plastic_bottle,bottle
```

In this example, both `colored_plastic_bottle` and `clear_plastic_bottle` will be updated to `bottle` in the new directory.

You can change `alwaysai.app.json` to run `label_converter.py`, specifying the input directory with the `--input_dir` flag. You can specify an output directory with the `--output_dir` flag, otherwise a default filename will be chosen for you and a new zip file will be created.