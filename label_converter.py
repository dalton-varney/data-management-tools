
import os
import zipfile
import csv
import xml.etree.ElementTree as ET
import argparse
import shutil


def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


def main(input_dir, output_dir):
    # Check for invalid input directory
    if not os.path.exists(input_dir):
        input_dir = os.path.join(os.getcwd(), input_dir)
        if not os.path.exists(input_dir):
            raise RuntimeError("Invalid input directory")

    if output_dir == "":
        input_suffix_ind = input_dir.rfind(".")
        output_dir = "{}_renamed".format(input_dir[:input_suffix_ind])

    print("Unzipping files...")
    if zipfile.is_zipfile(input_dir):
        print("Extracting " + input_dir + "...")
        with zipfile.ZipFile(input_dir, 'r') as zip:
            zip.extractall(output_dir)

    label_mappings = 'label_mappings.csv'
    with open(label_mappings) as f:
        label_data = [tuple(line) for line in csv.reader(f)]

    bad_labels = {bad: 0 for bad, _ in label_data}
    good_labels = {good: 0 for _, good in label_data}

    # create a new directory for the annotations at the level of the
    # folder that contains the annotation sets
    annotation_path = os.path.join(output_dir, "Annotations")

    print("Replacing Labels")

    # get all annotation files for a particular annotation task
    annotation_files = get_all_file_paths(annotation_path)

    # read in annotation data
    for annotation_file in annotation_files:
        print(annotation_file)
        tree = ET.parse(annotation_file)
        root = tree.getroot()

        # replace labels
        for bad, good in label_data:
            for obj in root.findall('object'):
                for name in obj.findall('name'):
                    if name.text == bad:
                        bad_labels[bad] += 1
                        good_labels[good] += 1
                        name.text = good

                        if good == 'omit':
                            root.remove(obj)

        tree.write(annotation_file)

    print("Program Finished")
    print("Label Stats:")
    print("___________________")
    print("\tOriginal Labels:")
    for label in bad_labels:
        print("\t\t" + str(label) + ": " + str(bad_labels[label]))

    print("\tNew Labels:")
    for label in good_labels:
        print("\t\t" + str(label) + ": " + str(good_labels[label]))

    print("Zipping files...")
    shutil.make_archive(
        "{}".format(output_dir), "zip", output_dir)
    shutil.rmtree(output_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='alwaysAI Label Converter Module')
    parser.add_argument(
            '--input_dir', type=str,
            help='The directory to augment; a zip file of an Annotation folder, in Pascal VOC, and a JPEGImages folder.')
    parser.add_argument(
                '--output_dir', type=str, default="",
                help='The directory to save the updated files to.')

    args = parser.parse_args()

    main(input_dir=args.input_dir, output_dir=args.output_dir)
