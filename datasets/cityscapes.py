import torch

from torch.utils.data import Dataset

import numpy as np
# import cv2
import os

train_dirs = [
    'jena/',
    'zurich/',
    'weimar/',
    'ulm/',
    'tubingen/',
    'stuttgart/',
    'strasbourg/',
    'monchengladbach/',
    'krefeld/',
    'hanover/',
    'hamburg/',
    'erfurt/',
    'dusseldorf/',
    'darmstadt/',
    'cologne/',
    'bremen/',
    'bochum/',
    'aachen/'
]

val_dirs = [
    'frankfurt/',
    'munster/',
    'lindau/'
]

test_dirs = [
    'berlin/',
    'bielefeld/',
    'bonn/',
    'leverkusen/',
    'mainz/',
    'munich/'
]

CLASS_MAPPING = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255,
    10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255,
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
    29: 255, 30: 255, 31: 16, 32: 17, 33: 18
}

# #{ Cityscapes

class Cityscapes(Dataset):
    def __init__(self, root_path, mode='train'):
        self.images_dir = root_path + '/leftImg8bit/'
        self.labels_dir = root_path + '/gtFine/'

        self.original_height = 1024
        self.original_width = 2048

        self.instances = []

        if mode == 'train':
            for train_dir in train_dirs:
                train_images_dir_path = self.images_dir + train_dir
                filenames = os.listdir(train_images_dir_path)

                for filename in filenames:
                    image_id = filename.split('_leftImg8bit.png')[0]

                    image_path = train_img_dir_path + filename

                    label_path = self.labels_dir + train_dir + image_id + '_gtFine_labelIds.png'

                    instance = {}
                    instance['image_path'] = image_path
                    print('Image path:', image_path)

                    instance['label_path'] = label_path
                    print('Label path:', label_path)

                    instance['image_id'] = image_id
                    print('Image ID:', image_id)

                    self.instances.append(instance)

        self.num_instances = len(self.instances)
        print('Number of instances:', self.num_instances)

# #}

if __name__=='__main__':

    home_dir = os.path.expanduser('~')
    root_dir = os.path.join(home_dir, '../app/data/cityscapes')
    train_dataset = Cityscapes(root_dir=root_dir)
