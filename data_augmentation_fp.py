import numpy as np
import util
import torch.utils.data as Data
import random
import torchvision.transforms.functional as TF
import torch
import pandas as pd


class FingerprintDataset(Data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, csv_file, img_width, img_height, pad_width, transform=None):
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
        util.read_bins_toCSV(root_dir, csv_file, img_width, img_height, True)

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

        image = util.read_bin(self.landmarks_frame.iloc[idx, 0], (self.width, self.height), True)
        bk = util.read_bin(self.landmarks_frame.iloc[idx, 1], (self.width, self.height), True)
        ipp = util.read_8bit_bin(self.landmarks_frame.iloc[idx, 2], (self.width, self.height), True)
        # diff = util.subtract(image, bk)
        util.mss_interpolation(image, self.width, self.height)
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

        image = util.read_bin(self.landmarks_frame.iloc[idx, 0], (self.width, self.height), True)
        bk = util.read_bin(self.landmarks_frame.iloc[idx, 1], (self.width, self.height), True)
        ipp_path = self.landmarks_frame.iloc[idx, 2]
        # diff = util.subtract(image, bk)
        util.mss_interpolation(image, self.width, self.height)
        diff = image

        # # normalize
        # diff = ((diff - np.mean(diff)) / np.std(diff)).astype('float32')

        # to uint8
        diff = ((diff - np.mean(diff)) / (np.max(diff) - np.min(diff))).astype('float32')

        diff = np.pad(diff, self.pad_width, 'reflect')
        diff = TF.to_tensor(diff)
        diff.unsqueeze_(0)
        return diff, ipp_path


if __name__ == '__main__':
    pass
