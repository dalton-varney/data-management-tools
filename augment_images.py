import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import os
import zipfile
import csv
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


class Augmenter:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.original_input_dir_name = None
        self.output_dir = None

    def get_updated_annotation(self, input_annotation, bounding_boxes):
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

    def write_augmented_files(
            self, aug_bbs, aug_image, aug_str, annotation_file):
        aug_image_path = "{}{}".format(
            os.path.join(
                self.image_path, self.i + aug_str), self.i_suffix)

        cv2.imwrite(aug_image_path, aug_image)
        aug_annotation = self.get_updated_annotation(self.tree, aug_bbs)
        aug_annotation.find('filename').text = "{}{}{}".format(
            self.i, aug_str, self.i_suffix)
        aug_annotation.write(
            os.path.join(
                self.new_annotation_path, annotation_file.replace(
                    ".xml", "{}.xml".format(aug_str))))

    def augment_images(
        self, aug_all, rotate_180, darken, rotate_90_darken,
            rotate_180_darken, brighten,
            rotate_brighten, blur, rotate_180_blur,
            rotate_270_darken, grayscale, rotate_90_grayscale,
            rotate_180_grayscale, grayscale_darken, grayscale_brighten,
            grayscale_blur, rotate_270_grayscale, zoom):
        # Check for invalid input directory
        if not os.path.exists(self.input_dir):
            input_dir = os.path.join(os.getcwd(), self.input_dir)
            if not os.path.exists(input_dir):
                raise RuntimeError("Invalid input directory")

        res_date = time.strftime('%Y.%m.%d')
        res_time = time.strftime('%H.%M.%S')

        input_suffix_ind = self.input_dir.rfind(".")
        self.original_input_dir_name = self.input_dir[:input_suffix_ind]
        self.output_dir = "{}-{}_augmented_{}".format(
            res_date, res_time, self.input_dir[:input_suffix_ind], )

        # Check for existing output directory
        if os.path.exists(os.path.join(os.getcwd(), self.output_dir)) or \
                os.path.exists(
                    os.path.join(os.getcwd(), self.output_dir, ".zip")):
            raise RuntimeError("Output directory already exists.")

        print("Unzipping files...")
        if zipfile.is_zipfile(self.input_dir):
            print("Extracting " + self.input_dir + "...")
            with zipfile.ZipFile(self.input_dir, 'r') as zip:
                zip.extractall(self.output_dir)

        # create the new directories for Annotations and JPEGImages
        self.new_annotation_path = os.path.join(self.output_dir, "Annotations")
        print("new annotation path: {}".format(self.new_annotation_path))
        self.new_image_directory = os.path.join(self.output_dir, 'JPEGImages')

        # get all annotation files for a particular annotation task
        annotation_files = get_all_file_paths(self.new_annotation_path)

        print("Augmenting images...")
        # read in annotation data one file at a time
        for annotation_file in annotation_files:
            # read in the annotation for the image
            # print(annotation_file)
            self.tree = ET.parse(annotation_file)
            root = self.tree.getroot()

            # make the new image path and name
            image_name = root.find('filename').text
            original_image_path = os.path.join(
                self.output_dir, 'JPEGImages', image_name)
            image = cv2.imread(original_image_path)
            self.image_path = self.new_image_directory

            # make a new annotation name and read in the annotation data
            image_suffix_ind = image_name.rfind(".")
            self.i = image_name[:image_suffix_ind]
            self.i_suffix = image_name[image_suffix_ind:]

            cv2.imwrite("{}{}".format(
                os.path.join(self.image_path, self.i), self.i_suffix), image)

            new_annotation_name = annotation_file[annotation_file.rfind(
                os.sep) + 1:]
            # write out the original image and annotation
            self.tree.find('filename').text = "{}{}".format(
                self.i, self.i_suffix)
            print(os.path.join(
                self.new_annotation_path, new_annotation_name))
            self.tree.write(os.path.join(
                self.new_annotation_path, new_annotation_name))

            # Now augment the images and update the bounding boxes
            # all the bounding boxes
            bbs = []
            for obj in root.findall('object'):
                for box in obj.findall('bndbox'):
                    x1 = float(box.find('xmin').text)
                    x2 = float(box.find('xmax').text)
                    y1 = float(box.find('ymin').text)
                    y2 = float(box.find('ymax').text)

                    bbs.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))

            # container for all the bounding boxes
            bbs = BoundingBoxesOnImage(bbs, image.shape)

            # flip the image 180
            if rotate_180 or aug_all:
                seq = iaa.Rot90(2)
                rot180_image, rot180_bbs = seq(image=image, bounding_boxes=bbs)
                self.write_augmented_files(
                    rot180_bbs,
                    rot180_image,
                    "_rotate_180",
                    new_annotation_name)

            # darken the image
            if darken or aug_all:
                seq = iaa.Multiply((0.7, 0.8))
                darkened_image, darkened_bbs = seq(
                    image=image, bounding_boxes=bbs)
                self.write_augmented_files(
                    darkened_bbs,
                    darkened_image,
                    "_darkened",
                    new_annotation_name)

            # flip 90 degrees and darken
            if rotate_90_darken or aug_all:
                seq = iaa.Multiply((0.7, 0.8))
                darkened_image, darkened_bbs = seq(
                    image=image, bounding_boxes=bbs)
                seq = iaa.Rot90(1)
                darkened_rot90_image, darkened_rot90_bbs = seq(
                    image=darkened_image, bounding_boxes=darkened_bbs)
                self.write_augmented_files(
                    darkened_rot90_bbs,
                    darkened_rot90_image,
                    "_darkened_rotate_90",
                    new_annotation_name)

            # flip 180 and darken
            if rotate_180_darken or aug_all:
                seq = iaa.Multiply((0.7, 0.8))
                darkened_image, darkened_bbs = seq(
                    image=image, bounding_boxes=bbs)
                seq = iaa.Rot90(2)
                dark_rot180_image, dark_rot180_bbs = seq(
                    image=darkened_image, bounding_boxes=darkened_bbs)
                self.write_augmented_files(
                    dark_rot180_bbs,
                    dark_rot180_image,
                    "_darkened_rotate_180",
                    new_annotation_name)

            # brighten the image
            if brighten or aug_all:
                seq = iaa.Multiply((1.4, 1.6))
                brightened_image, brightened_bbs_aug = seq(
                    image=image, bounding_boxes=bbs)
                self.write_augmented_files(
                    brightened_bbs_aug,
                    brightened_image,
                    "_brightened",
                    new_annotation_name)

            # flip 180 and brighten
            if rotate_brighten or aug_all:
                seq = iaa.Multiply((1.4, 1.6))
                brightened_image, brightened_bbs_aug = seq(
                    image=image, bounding_boxes=bbs)
                seq = iaa.Rot90(2)
                bright_rot180_image, brightened_rot180_bbs = seq(
                    image=brightened_image, bounding_boxes=brightened_bbs_aug)
                self.write_augmented_files(
                    brightened_rot180_bbs,
                    bright_rot180_image,
                    "_brightened_rotate_180",
                    new_annotation_name)

            # blur the image
            if blur or aug_all:
                seq = iaa.GaussianBlur(2)
                blurred_image, blurred_bbs_aug = seq(
                    image=image, bounding_boxes=bbs)
                self.write_augmented_files(
                    blurred_bbs_aug,
                    blurred_image,
                    "_blurred",
                    new_annotation_name)

            # flip 180 and blur
            if rotate_180_blur or aug_all:
                seq = iaa.GaussianBlur(2)
                blurred_image, blurred_bbs_aug = seq(
                    image=image, bounding_boxes=bbs)
                seq = iaa.Rot90(2)
                blurred_rot180_image, blurred_rot180_bbs = seq(
                    image=blurred_image, bounding_boxes=blurred_bbs_aug)
                self.write_augmented_files(
                    blurred_rot180_bbs,
                    blurred_rot180_image,
                    "_blurred_rotate_180",
                    new_annotation_name)

            # flip 270 and darken
            if rotate_270_darken or aug_all:
                seq = iaa.Multiply((0.7, 0.8))
                darkened_image, darkened_bbs = seq(
                    image=image, bounding_boxes=bbs)
                seq = iaa.Rot90(3)
                rot270_darkened_image, rot270_darkened_bbs = seq(
                    image=darkened_image, bounding_boxes=darkened_bbs)
                self.write_augmented_files(
                    rot270_darkened_bbs,
                    rot270_darkened_image,
                    "_rotate_270_darkened",
                    new_annotation_name)

            # - - - - GREYSCALE - - - - #
            # just grayscale
            if grayscale or aug_all:
                seq = iaa.color.ChangeColorspace("GRAY")
                gray_image, gray_bbs = seq(image=image, bounding_boxes=bbs)
                self.write_augmented_files(
                    gray_bbs,
                    gray_image,
                    "_grayscale",
                    new_annotation_name)

            # flip 90 and grayscale
            if rotate_90_grayscale or aug_all:
                seq = iaa.color.ChangeColorspace("GRAY")
                gray_image, gray_bbs = seq(image=image, bounding_boxes=bbs)
                seq = iaa.Rot90(1)
                rot90_gray_image, rot90_gray_bbs = seq(
                    image=gray_image, bounding_boxes=gray_bbs)
                self.write_augmented_files(
                    rot90_gray_bbs,
                    rot90_gray_image,
                    "_rotate_90_grayscale",
                    new_annotation_name)

            # flip 180 and grayscale
            if rotate_180_grayscale or aug_all:
                seq = iaa.color.ChangeColorspace("GRAY")
                gray_image, gray_bbs = seq(image=image, bounding_boxes=bbs)
                seq = iaa.Rot90(2)
                rot180_gray_image, rot180_gray_bbs = seq(
                    image=gray_image, bounding_boxes=gray_bbs)
                self.write_augmented_files(
                    rot180_gray_bbs,
                    rot180_gray_image,
                    "_rotate_180_grayscale",
                    new_annotation_name)

            # grayscale and darken
            if grayscale_darken or aug_all:
                seq = iaa.Multiply((0.7, 0.8))
                darkened_image, darkened_bbs = seq(
                    image=image, bounding_boxes=bbs)
                seq = iaa.color.ChangeColorspace("GRAY")
                dark_gray_image, dark_gray_bbs = seq(
                    image=darkened_image, bounding_boxes=darkened_bbs)
                self.write_augmented_files(
                    dark_gray_bbs,
                    dark_gray_image,
                    "_grayscale_darkened",
                    new_annotation_name)

            # grayscale and brighten
            if grayscale_brighten or aug_all:
                seq = iaa.Multiply((1.4, 1.6))
                brightened_image, brightened_bbs_aug = seq(
                    image=image, bounding_boxes=bbs)
                seq = iaa.color.ChangeColorspace("GRAY")
                bright_gray_image, bright_gray_bbs = seq(
                    image=brightened_image, bounding_boxes=brightened_bbs_aug)
                self.write_augmented_files(
                    bright_gray_bbs,
                    bright_gray_image,
                    "_grayscale_brightened",
                    new_annotation_name)

            # grayscale and blur
            if grayscale_blur or aug_all:
                seq = iaa.GaussianBlur(2)
                blurred_image, blurred_bbs_aug = seq(
                    image=image, bounding_boxes=bbs)
                seq = iaa.color.ChangeColorspace("GRAY")
                blurred_gray_image, blurred_gray_bbs = seq(
                    image=blurred_image, bounding_boxes=blurred_bbs_aug)
                self.write_augmented_files(
                    blurred_gray_bbs,
                    blurred_gray_image,
                    "_grayscale_blurred",
                    new_annotation_name)

            # flip 270 and grayscale
            if rotate_270_grayscale or aug_all:
                seq = iaa.color.ChangeColorspace("GRAY")
                gray_image, gray_bbs = seq(image=image, bounding_boxes=bbs)
                seq = iaa.Rot90(3)
                rot270_gray_image, rot270_gray_bbs = seq(
                    image=gray_image, bounding_boxes=gray_bbs)
                self.write_augmented_files(
                    rot270_gray_bbs,
                    rot270_gray_image,
                    "_rotate_270_grayscale",
                    new_annotation_name)

            # - - - - ZOOM - - - - #
            if zoom or aug_all:
                seq = iaa.Affine(scale={"x": (0.6, 1.0), "y": (0.6, 1.0)})
                zoomed_image, zoom_bbs = seq(image=image, bounding_boxes=bbs)
                self.write_augmented_files(
                    zoom_bbs,
                    zoomed_image,
                    "_zoomed",
                    new_annotation_name)

        print("Zipping files...")
        shutil.make_archive(
            "{}".format(self.output_dir), "zip", self.output_dir)
        print("Done.")
        shutil.rmtree(self.output_dir)

        # aug_zip = zipfile.ZipFile("{}.zip".format(self.output_dir), 'w')

        # for root, _, files in os.walk(self.output_dir):
        #     aug_zip.write(root)
        #     for filename in files:
        #         aug_zip.write(os.path.join(root, filename))
        # aug_zip.close()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='alwaysAI Augmentation Module')
        parser.add_argument(
                '--input_dir', type=str,
                help='The directory to augment; a zip file of an Annotation folder, in Pascal VOC, and a JPEGImages folder.')
        parser.add_argument('--all', action='store_true')
        parser.add_argument('--rotate_180', action='store_true')
        parser.add_argument('--darken', action='store_true'),
        parser.add_argument('--rotate_90_darken', action='store_true'),
        parser.add_argument('--rotate_180_darken', action='store_true')
        parser.add_argument('--brighten', action='store_true')
        parser.add_argument('--rotate_brighten', action='store_true')
        parser.add_argument('--blur', action='store_true')
        parser.add_argument('--rotate_180_blur', action='store_true')
        parser.add_argument('--rotate_270_darken', action='store_true')
        parser.add_argument('--grayscale', action='store_true')
        parser.add_argument('--rotate_90_grayscale', action='store_true')
        parser.add_argument(
            '--rotate_180_grayscale', action='store_true')
        parser.add_argument('--grayscale_darken', action='store_true')
        parser.add_argument('--grayscale_brighten', action='store_true')
        parser.add_argument('--grayscale_blur', action='store_true')
        parser.add_argument(
            '--rotate_270_grayscale', action='store_true')
        parser.add_argument('--zoom', action='store_true')

        args = parser.parse_args()
        print(args)

        augmenter = Augmenter(args.input_dir)
        augmenter.augment_images(
            aug_all=args.all,
            rotate_180=args.rotate_180,
            darken=args.darken,
            rotate_90_darken=args.rotate_90_darken,
            rotate_180_darken=args.rotate_180_darken,
            brighten=args.brighten,
            rotate_brighten=args.rotate_brighten,
            blur=args.blur,
            rotate_180_blur=args.rotate_180_blur,
            rotate_270_darken=args.rotate_270_darken,
            grayscale=args.grayscale,
            rotate_90_grayscale=args.rotate_90_grayscale,
            rotate_180_grayscale=args.rotate_180_grayscale,
            grayscale_darken=args.grayscale_darken,
            grayscale_brighten=args.grayscale_brighten,
            grayscale_blur=args.grayscale_blur,
            rotate_270_grayscale=args.rotate_270_grayscale,
            zoom=args.zoom
            )
    except RuntimeError as err:
        print(err)
