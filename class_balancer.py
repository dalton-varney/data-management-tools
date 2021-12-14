import argparse
import os
import shutil
import xml.etree.ElementTree as et
import edgeiq
import random


def _convert_xml_float_2_int(v):
    return int(round(float(v)))


class ClassBalancer:
    """
    Analyze and balance the class distribution of dataset available in PASCAL-VOC format.

    Note: Filenames must not contain "." other than for extension
    Typical usage::

        input_dir = <path_to_datset>
        balancer = ClassBalancer()
        class_distribrution = balancer.analyze_dataset(input_dir)

        balanced_distribution, move_list = balancer.get_balancing_stats(allowed_imbalance=0.10)

        balancer.partition_dataset(move_list)

    """
    def __init__(self):
        self.dict_fnames_class = {}
        self.class_distribution = {}

    def _parse_voc_annotations(self, input_dir):
        self.path_annotations = os.path.join(input_dir, 'Annotations')
        self.path_images = os.path.join(input_dir, 'JPEGImages')
        self.path_holdout_dir = os.path.join(input_dir, 'holdout')
        self.path_holdout_annotations = os.path.join(self.path_holdout_dir, 'Annotations')
        self.path_holdout_images = os.path.join(self.path_holdout_dir, 'JPEGImages')

        frame_predictions = {}
        all_predictions = []
        class_predictions = {}
        class_fnames = {}
        annotation_files = edgeiq.list_files(self.path_annotations, valid_exts=('.xml'))
        for f in annotation_files:
            tree = et.parse(f)

            fname = os.path.basename(f)
            if fname not in class_fnames.keys():
                class_fnames[fname] = {}

            root = tree.getroot()
            predictions = []
            for obj in root:
                if obj.tag != 'object':
                    continue

                for xml_box in obj:
                    if xml_box.tag == 'name':
                        label = xml_box.text
                        if label not in class_predictions.keys():
                            class_predictions[label] = []
                        if label not in class_fnames[fname].keys():
                            class_fnames[fname][label] = 1
                        else:
                            class_fnames[fname][label] += 1

                    if xml_box.tag != 'bndbox':
                        continue

                    box = edgeiq.BoundingBox(
                            start_x=_convert_xml_float_2_int(xml_box[0].text),
                            start_y=_convert_xml_float_2_int(xml_box[1].text),
                            end_x=_convert_xml_float_2_int(xml_box[2].text),
                            end_y=_convert_xml_float_2_int(xml_box[3].text))
                    predictions.append(
                            edgeiq.ObjectDetectionPrediction(box, 0, label, 0))
                    class_predictions[label].append(edgeiq.ObjectDetectionPrediction(box, 0, label, 0))

            all_predictions.extend(predictions)
            frame_predictions[fname] = predictions
            self.dict_fnames_class = class_fnames
        return frame_predictions, all_predictions, class_predictions

    def analyze_dataset(self, path):
        """
        Get class distribution for input dataset.

        :type path: string
        :param path: The path to dataset folder
        :returns: dictionary -- sorted class distribution of the dataset
        """
        frame_predictions, all_predictions, class_predictions = \
            self._parse_voc_annotations(path)
        self.class_distribution = {
            i: len(class_predictions[i]) for i in class_predictions.keys()}
        self.class_distribution = dict(
            sorted(self.class_distribution.items(), key=lambda x: x[1]))
        return self.class_distribution

    def get_balancing_stats(self, allowed_imbalance=0.00):
        """
        Understand the balancing flexibility and impact on the dataset.

        :type allowed_imbalance: float
        :param allowed_imbalance: normalized acceptable imbalance.
        :returns: dictionary -- best achievable class distribution
        :returns: list -- files that will be moved to achieve the balance
        """
        balancing_base = min(
            self.class_distribution.items(), key=lambda x: x[1])
        class_fnames_list = list(self.dict_fnames_class.items())
        total_images = len(class_fnames_list)

        balancing_classes = {}
        balanced_classes = {}

        allowed_thresh = 1+allowed_imbalance

        for label, count in self.class_distribution.items():
            if(count > allowed_thresh*balancing_base[1]):
                balancing_classes[label] = count
            else:
                balanced_classes[label] = count

        to_move = []
        processed_list = []
        while len(balancing_classes) != 0:
            if len(class_fnames_list) != 0:
                idx = random.randint(0,len(class_fnames_list)-1)
                frame_data = class_fnames_list[idx]
                class_fnames_list.pop(idx)
                processed_list.append(frame_data[0])

                if(
                    len((
                        set(balanced_classes.keys()) &
                        set(frame_data[1].keys()))) == 0):
                    to_update = {}

                    deletable_frame = True
                    for label, count in frame_data[1].items():
                        if(balancing_classes[label] <= allowed_thresh *
                                balancing_base[1]):
                            balanced_classes[label] = balancing_classes[label]
                            deletable_frame = False
                            break
                        else:
                            to_update[label] = count

                    if(deletable_frame):
                        to_move.append(frame_data[0])
                        for label, count in to_update.items():
                            balancing_classes[label] -= count
            else:
                new_class_distribution = {}
                new_class_distribution.update(balanced_classes)
                new_class_distribution.update(balancing_classes)
                print("BEST ACHIEVABLE BALANCE", new_class_distribution)
                print('NOTE: %d/%d images from your dataset are required to be moved to achieve this balancing result.' % (len(to_move), total_images))
                return new_class_distribution, to_move

        # If successfully achieved required balancing
        new_class_distribution = {}
        new_class_distribution.update(balanced_classes)
        new_class_distribution.update(balancing_classes)
        print("SUCCESFULLY ACHIEVED REQUIRED BALANCE", new_class_distribution)
        print('NOTE: %d/%d images from your dataset are required to be moved to achieve this balancing result.' % (len(to_move), total_images))
        return new_class_distribution, to_move

    def _create_required_dirs(self):
        if not os.path.isdir(self.path_holdout_dir):
            os.mkdir(self.path_holdout_dir)

        if not os.path.isdir(self.path_holdout_annotations):
            os.mkdir(self.path_holdout_annotations)

        if not os.path.isdir(self.path_holdout_images):
            os.mkdir(self.path_holdout_images)

    def partition_dataset(self, to_move, **kwargs):
        """
        Move the files required to balance the dataset.
        By default the files are moved to a "holdout" folder in the dataset directory.
        Output path can be modified using output_dir parameter.

        :type to_move: list
        :param to_move: list of images to be moved (output of get_balancing_stats())
        """
        self.path_holdout_dir = kwargs.get('output_dir', self.path_holdout_dir)
        self.path_holdout_annotations = os.path.join(
            self.path_holdout_dir, 'Annotations')
        self.path_holdout_images = os.path.join(
            self.path_holdout_dir, 'JPEGImages')

        self._create_required_dirs()

        for files in to_move:
            y, _exten = files.split('.')
            dest = shutil.move(
                self.path_annotations + os.sep + y + '.' + _exten,
                self.path_holdout_annotations)
            dest = shutil.move(
                self.path_images + os.sep + y + '.' + 'jpg',
                self.path_holdout_images)
        print("%d images and respective annotation files moved to %s" % (
            len(to_move), self.path_holdout_dir))


########### TESTING ##################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='alwaysAI Class Balancing Module')
    parser.add_argument(
                '--input_dir', type=str,
                help='The directory to analyze; a folder (unzipped) with Annotation folder, in Pascal VOC, and a JPEGImages folder.')
    parser.add_argument('--allowed_imbalance', type=float, default=0.00)
    parser.add_argument('--partition', action='store_true')
    parser.add_argument(
                '--output_dir', type=str,
                help='The directory to store files on partioning. A holdout folder in input_dir will be created and used if not specified.')
    args = parser.parse_args()
    print(args)

    balancer = ClassBalancer()
    class_distri = balancer.analyze_dataset(path=args.input_dir)
    print("Initial Class Distribution: ", class_distri)

    new_class_distri, move_list = balancer.get_balancing_stats(
        allowed_imbalance=args.allowed_imbalance)

    if args.partition is True:
        if args.output_dir is not None:
            balancer.partition_dataset(move_list, output_dir=args.output_dir)
        else:
            balancer.partition_dataset(move_list)
