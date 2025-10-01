import numpy as np

from pathlib import Path

from PIL import Image

import torch

from torch.utils.data import Dataset

import torchvision.transforms as transforms
import torchvision.transforms.functional as functional

# train_dirs = [
#     'jena/',
#     'zurich/',
#     'weimar/',
#     'ulm/',
#     'tubingen/',
#     'stuttgart/',
#     'strasbourg/',
#     'monchengladbach/',
#     'krefeld/',
#     'hanover/',
#     'hamburg/',
#     'erfurt/',
#     'dusseldorf/',
#     'darmstadt/',
#     'cologne/',
#     'bremen/',
#     'bochum/',
#     'aachen/'
# ]

train_dirs = [
   'darmstadt/',
]

# val_dirs = [
#     'frankfurt/',
#     'munster/',
#     'lindau/'
# ]

val_dirs = [
    'lindau/',
]

# test_dirs = [
#     'berlin/',
#     'bielefeld/',
#     'bonn/',
#     'leverkusen/',
#     'mainz/',
#     'munich/'
# ]

test_dirs = [
    'bonn/'
]

class_mapping = {
    0: 255,
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 0,
    8: 1,
    9: 255,
    10: 255,
    11: 2,
    12: 3,
    13: 4,
    14: 255,
    15: 255,
    16: 255,
    17: 5,
    18: 255,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: 255,
    30: 255,
    31: 16,
    32: 17,
    33: 18
}

# #{ DataTransformer

class DataTransformer:

    # (NOTE) Data augmentation mentioned in the article:
    #   - random cropping images (1024 x 1024);
    #   - random scaling in the range of 0.5 to 2.0;
    #   - random horizontal flipping.
    def __init__(self, partition, new_size, scale=(0.5, 2.0), horizontal_flip_probability=0.5):
        self.partition = partition
        self.new_size = new_size

        self.transform_normalize = transforms.Normalize(
            (0.28689529, 0.32491128, 0.28639529),
            (0.18696375, 0.19017339, 0.18725835)
        )

        if self.partition == 'train':
            self.horizontal_flip_probability = horizontal_flip_probability
            self.transform_random_resized_crop = transforms.RandomResizedCrop(size=new_size, scale=scale)
        else:
            self.transform_random_crop = transforms.RandomCrop(new_size)

    def __call__(self, image, mask):
        if self.partition == 'train':
            i, j, h, w = self.transform_random_resized_crop.get_params(
                image,
                scale=self.transform_random_resized_crop.scale,
                ratio=self.transform_random_resized_crop.ratio
            )

            image = functional.resized_crop(
                image,
                i, j, h, w,
                self.new_size,
                functional.InterpolationMode.BILINEAR
            )

            mask = functional.resized_crop(
                mask,
                i, j, h, w,
                self.new_size,
                functional.InterpolationMode.NEAREST
            )

            if torch.rand(1) < self.horizontal_flip_probability:
                image = functional.hflip(image)
                mask = functional.hflip(mask)
        else:
            i, j, h, w = self.transform_random_crop.get_params(image, output_size=self.new_size)

            image = functional.crop(image, i, j, h, w)
            mask = functional.crop(mask, i, j, h, w)

        image = transforms.ToTensor()(image)
        image = self.transform_normalize(image)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

# #}

# #{ Cityscapes

class Cityscapes(Dataset):

    def __init__(self, root, partition='train'):
        assert partition in ('train', 'val', 'test')

        self.root_path = Path(root)
        self.partition = partition
        self.images_dir_path = self.root_path / 'leftImg8bit' / self.partition
        self.masks_dir_path = self.root_path / 'gtFine' / self.partition

        if self.partition == 'train':
            self.cities_dirs = train_dirs
        elif self.partition == 'val':
            self.cities_dirs = val_dirs
        else:
            self.cities_dirs = test_dirs

        self.dataset = []

        self.data_transformer = DataTransformer(partition=self.partition, new_size=(1024, 1024))

        for city_dir in self.cities_dirs:
            city_dir_path = self.images_dir_path / city_dir
            image_filenames = [item.name for item in city_dir_path.iterdir() if item.is_file()]

            for image_filename in image_filenames:
                image_id = image_filename.split('_leftImg8bit.png')[0]
                image_path = city_dir_path / image_filename
                mask_path = self.masks_dir_path / city_dir / (image_id + '_gtFine_labelIds.png')

                data = {}
                data['image_path'] = image_path
                data['mask_path'] = mask_path
                data['image_id'] = image_id

                self.dataset.append(data)

        self.num_samples = len(self.dataset)


    def __getitem__(self, index):
        sample = self.dataset[index]
        image_path = sample['image_path']
        mask_path = sample['mask_path']

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # first, crop and resize both image and mask
        image, mask = self.data_transformer(image, mask)

        mask = np.array(mask, dtype=np.int64)
        mask = self.convert_mask(mask)
        mask = torch.from_numpy(mask).long()

        return (image, mask)

    def __len__(self):
        return self.num_samples

    def convert_mask(self, mask):
        new_mask = np.zeros_like(mask, dtype=np.uint8)
        for class_id, train_id in class_mapping.items():
            new_mask[mask == class_id] = train_id

        return new_mask

# #}

if __name__ == '__main__':

    home_path = Path.home()
    root_path = home_path / '../app/data/cityscapes'
    train_dataset = Cityscapes(root=root_path, partition='train')

    num_images = len(train_dataset)
    print(f'Number of images: {num_images}')
