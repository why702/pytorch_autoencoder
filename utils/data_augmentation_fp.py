import random
import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import torchvision.transforms.functional as TF

import utils.util as util
from utils.combine_genuines_fpdbindex import parse_genuines, parse_index, get_pair_info


class FingerprintDataset(Data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, csv_file, img_width, img_height, pad_width=0, RBS=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.width = img_width
        self.height = img_height
        self.pad_width = pad_width
        self.RBS = RBS
        self.low_endian = not RBS
        util.read_bins_toCSV(root_dir, csv_file, img_width, img_height, RBS=RBS, GOOD=False)

        self.landmarks_frame = pd.read_csv(csv_file)
        self.size = self.landmarks_frame.shape[0]
        self.root_dir = root_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def transform(self, image, label):
        # # Resize
        # resize = transforms.Resize(size=(520, 520))
        # image = resize(image)
        # label = resize(label)
        #
        # # Random crop
        # i, j, h, w = transforms.RandomCrop.get_params(
        #     image, output_size=(512, 512))
        # image = TF.crop(image, i, j, h, w)
        # label = TF.crop(label, i, j, h, w)

        image = TF.to_pil_image(image)
        label = TF.to_pil_image(label)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

        # Transform to tensor
        image = TF.to_tensor(image)
        label = TF.to_tensor(label)

        # image = TF.normalize(image, mean=(0,), std=(1,))
        # label = TF.normalize(label, mean=(0,), std=(1,))
        return image, label

    def __getitem__(self, idx, trans=True):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = util.read_bin(self.landmarks_frame.iloc[idx, 0], (self.width, self.height), self.low_endian)
        bk = util.read_bin(self.landmarks_frame.iloc[idx, 1], (self.width, self.height), self.low_endian)
        ipp = util.read_8bit_bin(self.landmarks_frame.iloc[idx, 2], (self.width, self.height))
        # diff = util.subtract(image, bk)
        util.mss_interpolation(image.astype('float32'), self.width, self.height)
        # plt.imshow(image)
        # plt.show()
        diff = image

        # # normalize
        # diff = ((diff - np.mean(diff)) / np.std(diff)).astype('float32')
        # ipp = ((ipp - np.mean(ipp)) / np.std(ipp)).astype('float32')

        # to uint8
        diff = ((diff - np.min(diff)) / (np.max(diff) - np.min(diff))).astype('float32')
        ipp = ipp.astype('float32')

        diff = np.pad(diff, self.pad_width, 'reflect')
        ipp = np.pad(ipp, self.pad_width, 'reflect')
        if trans:
            diff, ipp = self.transform(diff, ipp)

        return diff, ipp

    def get_img_path(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = util.read_bin(self.landmarks_frame.iloc[idx, 0], (self.width, self.height), self.low_endian)
        bk = util.read_bin(self.landmarks_frame.iloc[idx, 1], (self.width, self.height), self.low_endian)
        ipp_path = self.landmarks_frame.iloc[idx, 2]
        # diff = util.subtract(image, bk)
        util.mss_interpolation(image.astype('float32'), self.width, self.height)
        diff = image

        # # normalize
        # diff = ((diff - np.mean(diff)) / np.std(diff)).astype('float32')

        # to uint8
        diff = ((diff - np.mean(diff)) / (np.max(diff) - np.min(diff))).astype('float32')

        diff = np.pad(diff, self.pad_width, 'reflect')
        diff = TF.to_tensor(diff)
        diff.unsqueeze_(0)
        return diff, ipp_path


class PerfDataset(Data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, gen_file, index_file, csv_file, img_width, img_height, pad_width=0, RBS=False, PI=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.width = img_width
        self.height = img_height
        self.pad_width = pad_width
        self.RBS = RBS
        self.low_endian = not RBS
        self.PI = PI

        gen_data0 = parse_genuines(gen_file)
        index_data0 = parse_index(index_file)
        root_dir = os.path.dirname(index_file)
        get_pair_info(gen_data0, index_data0, root_dir, csv_file)

        self.landmarks_frame = pd.read_csv(csv_file)
        self.size = self.landmarks_frame.shape[0]
        print("Get {} pairs of image".format(self.size))

    def __len__(self):
        return len(self.landmarks_frame)

    def transform(self, image, label):
        # # Resize
        # resize = transforms.Resize(size=(520, 520))
        # image = resize(image)
        # label = resize(label)
        #
        # # Random crop
        # i, j, h, w = transforms.RandomCrop.get_params(
        #     image, output_size=(512, 512))
        # image = TF.crop(image, i, j, h, w)
        # label = TF.crop(label, i, j, h, w)

        image = TF.to_pil_image(image)
        label = TF.to_pil_image(label)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

        # Transform to tensor
        image = TF.to_tensor(image)
        label = TF.to_tensor(label)

        # image = TF.normalize(image, mean=(0,), std=(1,))
        # label = TF.normalize(label, mean=(0,), std=(1,))
        return image, label

    def __getitem__(self, idx, trans=True):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        enroll_ipp_path = self.landmarks_frame.iloc[idx, 7]
        verify_ipp_path = self.landmarks_frame.iloc[idx, 8]
        score = self.landmarks_frame.iloc[idx, 3]

        enroll_raw_path = None
        verify_raw_path = None
        if self.RBS:
            enroll_raw_path = enroll_ipp_path.replace("image_bin", "image_raw")
            verify_raw_path = verify_ipp_path.replace("image_bin", "image_raw")
        else:
            enroll_raw_path = enroll_ipp_path.replace("_Img8b_", "_Img16b_")
            verify_raw_path = verify_ipp_path.replace("_Img8b_", "_Img16b_")

        enroll_raw_path = enroll_raw_path.replace(".png", ".bin")
        verify_raw_path = verify_raw_path.replace(".png", ".bin")

        enroll_raw = util.read_bin(enroll_raw_path, (self.width, self.height), self.low_endian)
        verify_raw = util.read_bin(verify_raw_path, (self.width, self.height), self.low_endian)
        enroll_ipp = util.read_8bit_bin(enroll_ipp_path, (self.width, self.height))
        verify_ipp = util.read_8bit_bin(verify_ipp_path, (self.width, self.height))

        if self.PI:
            util.mss_interpolation(enroll_raw.astype('float32'), self.width, self.height)
            util.mss_interpolation(verify_raw.astype('float32'), self.width, self.height)

        # to uint8
        enroll_raw = ((enroll_raw - np.min(enroll_raw)) / (np.max(enroll_raw) - np.min(enroll_raw))).astype('float32')
        verify_raw = ((verify_raw - np.min(verify_raw)) / (np.max(verify_raw) - np.min(verify_raw))).astype('float32')
        enroll_ipp = enroll_ipp.astype('float32')
        verify_ipp = verify_ipp.astype('float32')

        # padding
        enroll_raw = np.pad(enroll_raw, self.pad_width, 'reflect')
        verify_raw = np.pad(verify_raw, self.pad_width, 'reflect')
        enroll_ipp = np.pad(enroll_ipp, self.pad_width, 'reflect')
        verify_ipp = np.pad(verify_ipp, self.pad_width, 'reflect')

        # transform
        if trans:
            enroll_raw, enroll_ipp = self.transform(enroll_raw, enroll_ipp)
            verify_raw, verify_ipp = self.transform(verify_raw, verify_ipp)

        return enroll_raw, enroll_ipp, verify_raw, verify_ipp, score


if __name__ == '__main__':
    pass
