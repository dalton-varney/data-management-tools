import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import os
import zipfile
import copy as cp
import xml.etree.ElementTree as ET
import argparse
import time
import shutil


def get_all_file_paths(directory):
    """
    Gets all the files in the specified input directory
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


def get_updated_annotation(input_annotation, bounding_boxes):
    """
    Function to update the annotation's bounding boxes
    :param input_annotation: the original annotation read in
    :param bounding_boxes: the augmented bounding boxes
    :return: the updated annotation in xml format
    """
    annotation = cp.deepcopy(input_annotation)
    count = 0
    root = annotation.getroot()

    for obj in root.findall('object'):
        for box in obj.findall('bndbox'):
            box.find('xmin').text = str(bounding_boxes[count].x1)
            box.find('xmax').text = str(bounding_boxes[count].x2)
            box.find('ymin').text = str(bounding_boxes[count].y1)
            box.find('ymax').text = str(bounding_boxes[count].y2)
            count += 1
    return annotation


def main(input_dir, output_dir, sample):

    if not os.path.exists(input_dir):
        input_dir = os.path.join(os.getcwd(), input_dir)
        if not os.path.exists(input_dir):
            raise RuntimeError("Invalid input directory")

    res_date = time.strftime('%Y.%m.%d')
    res_time = time.strftime('%H.%M.%S')

    input_suffix_ind = input_dir.rfind(".")
    if output_dir is None:
        output_dir = "{}-{}_TEST_{}".format(
            res_date, res_time, input_dir[:input_suffix_ind])

    # Check for existing output directory
    if os.path.exists(os.path.join(os.getcwd(), output_dir)) or \
            os.path.exists(
                os.path.join(os.getcwd(), output_dir, ".zip")):
        raise RuntimeError("Output directory already exists.")

    print("Unzipping files...")
    if zipfile.is_zipfile(input_dir):
        print("Extracting " + input_dir + "...")
        with zipfile.ZipFile(input_dir, 'r') as zip:
            zip.extractall(output_dir)
    # create the new directories for Annotations and JPEGImages
    new_annotation_path = os.path.join(output_dir, "Annotations")
    new_image_directory = os.path.join(output_dir, 'JPEGImages')

    test_image_directory = os.path.join(output_dir, 'TestImages')
    os.mkdir(test_image_directory)

    # get all annotation files for a particular annotation task
    annotation_files = get_all_file_paths(new_annotation_path)

    # get all annotation files for a particular annotation task
    if len(annotation_files) == 0:
        raise RuntimeError(
            "No such folder Annotations, invalid folder structure.")

    sample = sample if len(annotation_files) > sample else len(
        annotation_files)
    print("{} total annotation pairs present, testing {} pairs.".format(
        len(annotation_files), int(len(annotation_files)/sample)))

    print("Testing images...")
    count = 0
    # read in annotation data one file at a time
    for annotation_file in annotation_files:
        if count % sample == 0:
            print(annotation_file)
            # read in the annotation for the image
            tree = ET.parse(annotation_file)
            root = tree.getroot()

            # make the new image path and name
            image_name = root.find('filename').text
            original_image_path = os.path.join(new_image_directory, image_name)

            image = cv2.imread(original_image_path)

            if image is None:
                raise RuntimeError("Invalid image path {}".format(
                    original_image_path))

            for obj in root.findall('object'):
                label = obj.find('name').text
                for box in obj.findall('bndbox'):
                    x1 = int(float(box.find('xmin').text))
                    x2 = int(float(box.find('xmax').text))
                    y1 = int(float(box.find('ymin').text))
                    y2 = int(float(box.find('ymax').text))

                    start = (x1, y1)
                    end = (x2, y2)

                    if start is None or end is None:
                        raise RuntimeError(
                            "Invalid annotation box for file {}".format(
                                annotation_file))

                    else:
                        image_suffix_ind = image_name.rfind(".")
                        i = image_name[:image_suffix_ind]
                        i_suffix = image_name[image_suffix_ind:]
                        new_image_path = "{}{}{}{}{}".format(
                            test_image_directory,
                            os.sep,
                            i,
                            "_test_box",
                            i_suffix)
                        new_image = cv2.rectangle(
                            image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        new_image = cv2.putText(
                            new_image, label, (x2 + 5, y2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 0, 0), 1)
                        try:
                            cv2.imwrite(new_image_path, new_image)
                        except cv2.error:
                            print("ERROR")
                            print(new_image_path)

        count += 1
    shutil.rmtree(new_annotation_path)
    shutil.rmtree(new_image_directory)
    print("Tested {} images. Available in {}.".format(
        int(len(annotation_files)/sample), new_image_directory))
    print("Done.")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description='alwaysAI Annotation Check Module')
        parser.add_argument(
                '--input_dir', type=str,
                help='The directory to augment; a zip file of an Annotation folder, in Pascal VOC, and a JPEGImages folder.')
        parser.add_argument(
                '--output_dir', type=str,
                help='The directory to save the augmented images to.')
        parser.add_argument(
                '--sample', type=int, default=1,
                help='The sample rate to test annotations.')

        args = parser.parse_args()
        main(args.input_dir, args.output_dir, args.sample)
    except RuntimeError as err:
        print(err)
