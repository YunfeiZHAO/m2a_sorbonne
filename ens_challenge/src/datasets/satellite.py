"""
Classes and functions to handle data
"""
import os
import logging
import random

import numpy as np
import pandas as pd

from tifffile import TiffFile

import torch
from torch.utils.data import Dataset

from torch.utils.tensorboard import SummaryWriter


class LandCoverData():
    """Class to represent the S2GLC Land Cover Dataset for the challenge,
    with useful metadata and statistics.
    """
    # image size of the images and label masks
    IMG_SIZE = 256
    # the images are RGB+NIR (4 channels)
    N_CHANNELS = 4
    # we have 9 classes + a 'no_data' class for pixels with no labels (absent in the dataset)
    N_CLASSES = 10
    CLASSES = [
        'no_data',
        'clouds',
        'artificial',
        'cultivated',
        'broadleaf',
        'coniferous',
        'herbaceous',
        'natural',
        'snow',
        'water'
    ]
    # classes to ignore because they are not relevant. "no_data" refers to pixels without
    # a proper class, but it is absent in the dataset; "clouds" class is not relevant, it
    # is not a proper land cover type and images and masks do not exactly match in time.
    IGNORED_CLASSES_IDX = [0, 1]

    # The training dataset contains 18491 images and masks
    # The test dataset contains 5043 images and masks
    TRAINSET_SIZE = 18491
    TESTSET_SIZE = 5043

    # for visualization of the masks: classes indices and RGB colors
    CLASSES_COLORPALETTE = {
        0: [0,0,0],
        1: [255,25,236],
        2: [215,25,28],
        3: [211,154,92],
        4: [33,115,55],
        5: [21,75,35],
        6: [118,209,93],
        7: [130,130,130],
        8: [255,255,255],
        9: [43,61,255]
        }
    CLASSES_COLORPALETTE = {c: np.asarray(color) for (c, color) in CLASSES_COLORPALETTE.items()}

    # statistics
    # the pixel class counts in the training set
    TRAIN_CLASS_COUNTS = np.array(
        [0, 20643, 60971025, 404760981, 277012377, 96473046, 333407133, 9775295, 1071, 29404605]
    )
    # the minimum and maximum value of image pixels in the training set
    TRAIN_PIXELS_MIN = 1
    TRAIN_PIXELS_MAX = 24356


class SatelliteDataset(Dataset):
    def __init__(self, index_txt_path, images_folder_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/dataset/',
                 train_label_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/train_labels.csv',
                 type='train'):
        """ Initialisation of dataset
        :param index_txt_path: The path to the csv file include the corresponding data index
        :param images_folder_path: The root path of the folder include the images
        """
        self.writer = SummaryWriter(log_dir='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/visualisation/')
        # load index from csv file
        self.data_index = pd.read_csv(index_txt_path)
        self.train_label = pd.read_csv(train_label_path, index_col=0, sep=',')
        self.images_folder_path = images_folder_path
        self.type = type

        # existence of image verification
        if self.type == 'train' or self.type == 'val':
            existence_image = [os.path.isfile(os.path.join(self.images_folder_path, 'train/images', str(*i) + '.tif'))
                               for i in self.data_index.values]
            if not existence_image:
                raise RuntimeError(f'No input file found in {self.images_folder_path}/train/images, make sure you put your images there')

            existence_mask = [os.path.isfile(os.path.join(self.images_folder_path, 'train/masks', str(*i) + '.tif'))
                               for i in self.data_index.values]
            if not existence_mask:
                raise RuntimeError(f'No input file found in {self.images_folder_path}/train/masks, make sure you put your images there')
            logging.info(f'Creating dataset with {len(self.data_index)} examples')
        else:
            existence_image = [os.path.isfile(os.path.join(self.images_folder_path, 'test/images', str(*i) + '.tif'))
                               for i in self.data_index.values]
            if not existence_image:
                raise RuntimeError(f'No input file found in {self.images_folder_path}/test/images, make sure you put your images there')
            logging.info(f'Creating dataset with {len(self.data_index)} examples')

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, index):
        sample_id = int(self.data_index.loc[index].sample_id)
        im_name = str(sample_id) + '.tif'
        if self.type == 'train':
            image = TiffFile(os.path.join(self.images_folder_path, 'train/images', im_name)).asarray()
            mask = TiffFile(os.path.join(self.images_folder_path, 'train/masks', im_name)).asarray()
            # add channel dimension to mask: (256, 256, 1)
            image, mask = self.train_processing(image, mask)  # image: C,H,W    mask: H, W
            label = torch.from_numpy(self.train_label.loc[sample_id].values.astype(np.float32))
            return {'image_id': im_name, 'image': image, 'mask': mask, 'label': label}
        elif self.type == 'test':
            # for evaluation
            image = TiffFile(os.path.join(self.images_folder_path, 'test/images', im_name)).asarray()
            image = self.test_processing(image)
            return {'image_id': torch.tensor(self.data_index.loc[index].values[0]), 'image': image}
        elif self.type == 'val':
            image = TiffFile(os.path.join(self.images_folder_path, 'train/images', im_name)).asarray()
            mask = TiffFile(os.path.join(self.images_folder_path, 'train/masks', im_name)).asarray()
            image = self.test_processing(image)
            mask = torch.from_numpy(mask)
            label = torch.from_numpy(self.train_label.loc[sample_id].values.astype(np.float32))
            return {'image_id': torch.tensor(self.data_index.loc[index].values[0]), 'image': image,
                    'mask': mask, 'label': label}

    def train_processing(self, image, mask):
        """Pass to tensor, with normalisation and data augmentation"""
        image = torch.from_numpy(image.astype(np.float32))
        mask = torch.from_numpy(mask)
        # rescale the pixel values of the images between 0.0 and 0.1
        ################## Do we need to standarize it
        image = image / LandCoverData.TRAIN_PIXELS_MAX
        process = Compose([
            RandomFlipRotation()
        ])
        return process(image, mask)

    def test_processing(self, image):
        image = torch.from_numpy(image.astype(np.float32))
        image = image / LandCoverData.TRAIN_PIXELS_MAX
        return image.permute(2, 0, 1)

    def visualisation(self, index, display_min=50, display_max=3000):
        im_name = str(self.data_index.loc[index].values[0]) + '.tif'
        image = TiffFile(os.path.join(self.images_folder_path, 'train/images', im_name)).asarray()
        image = image[..., 0:3]
        mask = TiffFile(os.path.join(self.images_folder_path, 'train/masks', im_name)).asarray()

        image_batch = torch.zeros((8, 256, 256, 3))
        mask_batch = torch.zeros((8, 256, 256, 3))

        # image processing
        image = image.clip(display_min, display_max)
        image = image/1000  # divide by empirical mean
        img = torch.as_tensor(image.clip(0, 1), dtype=torch.float)
        image_batch[0] = img

        # mask processing
        show_mask = np.empty((*mask.shape, 3))
        for c, color in LandCoverData.CLASSES_COLORPALETTE.items():
            show_mask[mask == c, :] = color/255
        show_mask = torch.as_tensor(show_mask, dtype=torch.float)
        mask_batch[0] = show_mask

        # data augmentation #this is stupid, need to be changed permute at the end
        image_batch[1] = torch.rot90(img, -1)
        mask_batch[1] = torch.rot90(show_mask, -1)

        image_batch[2] = torch.rot90(img, 2)
        mask_batch[2] = torch.rot90(show_mask, 2)

        image_batch[3] = torch.rot90(img, 1)
        mask_batch[3] = torch.rot90(show_mask, 1)

        image_batch[4] = torch.flipud(img)
        mask_batch[4] = torch.flipud(show_mask)

        image_batch[5] = torch.rot90(img.flipud(), -1)
        mask_batch[5] = torch.rot90(show_mask.flipud(), -1)

        image_batch[6] = torch.rot90(img.flipud(), 2)
        mask_batch[6] = torch.rot90(show_mask.flipud(), 2)

        image_batch[7] = torch.rot90(img.flipud(), 1)
        mask_batch[7] = torch.rot90(show_mask.flipud(), 1)

        self.writer.add_images(f'{im_name}_image', image_batch.permute(0, 3, 1, 2))
        self.writer.add_images(f'{im_name}_mask', mask_batch.permute(0, 3, 1, 2))
        self.writer.close()
        return image, show_mask


class RandomFlipRotation(object):
    def __call__(self, img, mask):
        """flip and rotation for data augmentation
        :param img: H, W, C
        :param mask: H, W
        :return: img: C, H, W   mask: H, W
        """
        r = np.random.random()
        if r < 0.125:
            pass
        elif r < 0.25:
            # clockwise 90
            img = torch.rot90(img, -1)
            mask = torch.rot90(mask, -1)
        elif r < 0.375:
            # clockwise 180
            img = torch.rot90(img, 2)
            mask = torch.rot90(mask, 2)
        elif r < 0.5:
            # anticlockwise 90
            img = torch.rot90(img, 1)
            mask = torch.rot90(mask, 1)
        elif r < 0.625:
            # updown reversal
            img = torch.flipud(img)
            mask = torch.flipud(mask)
        elif r < 0.75:
            # updown reversal + clockwise 90
            img = torch.rot90(img.flipud(), -1)
            mask = torch.rot90(mask.flipud(), -1)
        elif r < 0.875:
            # updown reversal + clockwise 180
            img = torch.rot90(img.flipud(), 2)
            mask = torch.rot90(mask.flipud(), 2)
        elif r < 1:
            # updown reversal + anticlockwise 90
            img = torch.rot90(img.flipud(), 1)
            mask = torch.rot90(mask.flipud(), 1)
        return img.permute(2, 0, 1), mask


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


def split_dataset(origin_csv, train_path, val_path, ratio=0.8):
    data_index = pd.read_csv(origin_csv)
    mask = np.random.rand(len(data_index)) < ratio
    train = data_index[mask]
    val = data_index[~mask]
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)


# split_dataset('./train_images.csv', './train_index.csv', './val_index.csv', ratio=0.8)
data = SatelliteDataset('/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/train_index.csv',
                        images_folder_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/dataset/',
                        type='train')
sample = data.__getitem__(1)
image, mask = data.visualisation(20)

