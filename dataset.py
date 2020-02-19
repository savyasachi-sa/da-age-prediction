import csv
from shutil import copy2
from os import listdir
from os.path import isfile, join

# #Using dataset.py
# from dataset import *
# src = '/Users/vulcan/da-age-prediction/Data/utk/'
# ethnicity = {
#     "source" : 0,
#     "target" : 1,
# }
# source_val_split_perc = 80
# img_location = '/da-age-prediction/Data/'
# _dataset = dataset(src, ethnicity, source_val_split_perc, img_location)


class dataset():
    def __init__(self, src, ethnicity, source_val_split_perc, img_location):
        self.img_location = img_location

        source_ethnicity = ethnicity['source']
        target_ethnicity = ethnicity['target']
        source_dataset_path = src + str(source_ethnicity)
        target_dataset_path = src + str(target_ethnicity)

        source_img_files = [f for f in listdir(source_dataset_path) if isfile(join(source_dataset_path, f))]
        target_img_files = [f for f in listdir(target_dataset_path) if isfile(join(target_dataset_path, f))]

        train_end_idx = int(len(source_img_files) * source_val_split_perc * .01)
        fieldnames = ['images', 'labels']
        with open('./Data/source_train.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fieldnames)
            writer.writerows([self.get_location_image(source_ethnicity, img), self.get_age(img)] for img in source_img_files[0:train_end_idx])
        with open('./Data/source_validation.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fieldnames)
            writer.writerows([self.get_location_image(source_ethnicity, img), self.get_age(img)] for img in source_img_files[train_end_idx:len(source_img_files)])
        with open('./Data/target_train.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fieldnames)
            writer.writerows([self.get_location_image(target_ethnicity, img), self.get_age(img)] for img in target_img_files[0:len(target_img_files)])

    def get_location_image(self, ethnicity, img):
        return self.img_location + str(ethnicity) + '/' + img

    def get_age(self, img):
        return img.split('_')[0]
