import csv
from os import listdir
from os.path import isfile, join
from config import *


class DatasetDesign():
    def __init__(self, ethnicity, source_val_split_perc):
        self.img_location = DATASET_ROOT_DIRECTORY

        source_ethnicity = ethnicity['source']
        target_ethnicity = ethnicity['target']
        source_dataset_path = DATASET_ROOT_DIRECTORY + str(source_ethnicity)
        target_dataset_path = DATASET_ROOT_DIRECTORY + str(target_ethnicity)

        source_img_files = [f for f in listdir(source_dataset_path) if isfile(join(source_dataset_path, f))]
        target_img_files = [f for f in listdir(target_dataset_path) if isfile(join(target_dataset_path, f))]

        source_img_files = source_img_files[0:5]
        target_img_files = target_img_files[0:5]

        train_end_idx = int(len(source_img_files) * source_val_split_perc * .01)
        fieldnames = ['images', 'labels']
        with open(SOURCE_TRAIN_PATH, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fieldnames)
            writer.writerows([self.get_location_image(source_ethnicity, img), self.get_age(img)] for img in
                             source_img_files[0:train_end_idx])
        with open(SOURCE_VAL_PATH, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fieldnames)
            writer.writerows([self.get_location_image(source_ethnicity, img), self.get_age(img)] for img in
                             source_img_files[train_end_idx:len(source_img_files)])
        with open(TARGET_TRAIN_PATH, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fieldnames)
            writer.writerows([self.get_location_image(target_ethnicity, img), self.get_age(img)] for img in
                             target_img_files[0:len(target_img_files)])

    def get_location_image(self, ethnicity, img):
        return self.img_location + str(ethnicity) + '/' + img

    def get_age(self, img):
        return img.split('_')[0]
