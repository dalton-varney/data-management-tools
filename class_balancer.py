import os
import shutil
import xml.etree.ElementTree as et
import edgeiq
import random


class ClassBalancer:
    def __init__(self, input_dir):
        self.path_annotations = os.path.join(input_dir, 'Annotations')
        self.path_images = os.path.join(input_dir, 'JPEGImages')
        self.path_holdout_dir = os.path.join(input_dir, 'holdout')
        self.path_holdout_annotations = os.path.join(
            self.path_holdout_dir, 'Annotations')
        self.path_holdout_images = os.path.join(
            self.path_holdout_dir, 'JPEGImages')
        self.dict_fnames_class = {}
        self.class_distribution = {}

    def convert_xml_float_2_int(self, v):
        return int(round(float(v)))

    def parse_voc_annotations(self):
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
                            start_x=self.convert_xml_float_2_int(xml_box[0].text),
                            start_y=self.convert_xml_float_2_int(xml_box[1].text),
                            end_x=self.convert_xml_float_2_int(xml_box[2].text),
                            end_y=self.convert_xml_float_2_int(xml_box[3].text))
                    predictions.append(
                            edgeiq.ObjectDetectionPrediction(box, 0, label, 0))
                    class_predictions[label].append(
                        edgeiq.ObjectDetectionPrediction(box, 0, label, 0))

            all_predictions.extend(predictions)
            frame_predictions[fname] = predictions
            self.dict_fnames_class = class_fnames
        return frame_predictions, all_predictions, class_predictions

    def get_dataset_stats(self):
        frame_predictions, all_predictions, class_predictions = self.parse_voc_annotations()
        self.class_distribution = {i: len(class_predictions[i]) for i in class_predictions.keys()} 
        return self.class_distribution

    def get_balancing_stats(self, balance_threshold=0.00):
        balancing_base = min(
            self.class_distribution.items(), key=lambda x: x[1])
        class_fnames_list = list(self.dict_fnames_class.items())
        # print("Class fnames list length" , len(class_fnames_list))

        balancing_classes = {}
        balanced_classes = {}

        allowed_thresh = 1+balance_threshold

        for k, v in self.class_distribution.items():
            if(v > allowed_thresh*balancing_base[1]):
                balancing_classes[k] = v
            else:
                balanced_classes[k] = v

        to_move_list = []
        processed_list = []
        while len(balancing_classes) != 0:
            if len(class_fnames_list) != 0:
                idx = random.randint(0, len(class_fnames_list)-1)
                frame_data = class_fnames_list[idx]
                class_fnames_list.pop(idx)
                processed_list.append(frame_data[0])

                if(len((set(balanced_classes.keys()) & set(frame_data[1].keys()))) == 0):
                    # print("Can think forward to delete")
                    to_update = {}

                    deletable_frame = True
                    for k, v in frame_data[1].items():
                        if(balancing_classes[k] <= allowed_thresh * balancing_base[1]):
                            balanced_classes[k] = balancing_classes[k]
                            deletable_frame = False
                            break
                        else:
                            to_update[k] = v

                    if(deletable_frame):
                        to_move_list.append(frame_data[0])
                        for k, v in to_update.items():
                            balancing_classes[k] -= v
                # else:
                    # print("CAn't delete this")
            else:
                new_class_distribution = {}
                new_class_distribution.update(balanced_classes)
                new_class_distribution.update(balancing_classes)
                print("BEST ACHIEVABLE BALANCE", new_class_distribution)
                return new_class_distribution, to_move_list

#If successfully achieved required balancing 
        new_class_distribution = {}
        new_class_distribution.update(balanced_classes)
        new_class_distribution.update(balancing_classes)
        print("SUCCESFULLY ACHIEVED REQUIRED BALANCE", new_class_distribution)
        return new_class_distribution, to_move_list

    def _create_required_dirs(self):
        if not os.path.isdir(self.path_holdout_dir):
            os.mkdir(self.path_holdout_dir)

        if not os.path.isdir(self.path_holdout_annotations):
            os.mkdir(self.path_holdout_annotations)

        if not os.path.isdir(self.path_holdout_images):
            os.mkdir(self.path_holdout_images)

    def partition_balanced_dataset(self, to_move_list, **kwargs):
        self.path_holdout_dir = kwargs.get('output_dir', self.path_holdout_dir)
        self.path_holdout_annotations = os.path.join(
            self.path_holdout_dir, 'Annotations')
        self.path_holdout_images = os.path.join(
            self.path_holdout_dir, 'JPEGImages')

        self._create_required_dirs()

        for i in to_move_list:
            y, _exten = i.split('.')
            dest = shutil.move(
                os.path.join(
                    self.path_annotations, os.sep, y, '.' + _exten, self.path_holdout_annotations))
            dest = shutil.move(
                os.path.join(self.path_images, os.sep, y,'.', 'jpg',self.path_holdout_images))


########### TESTING ##################

input_dir = 'dataset_sample_584-2/'  # TODO: replace with your input directory
balancer = ClassBalancer(input_dir)
class_distri = balancer.get_dataset_stats()
print("Initial Class Distribution", class_distri)

new_class_distri, move_list = balancer.get_balancing_stats()

print("new dist", new_class_distri)
print("To be moved list", move_list)
