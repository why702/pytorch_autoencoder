import os
import random
import threading
from queue import Queue

import numpy as np
import pandas as pd
import progressbar
import torch
import torch.utils.data as Data
import torchvision.transforms.functional as TF
from PIL import Image

import utils.util as util
from utils.combine_genuines_fpdbindex import parse_genuines, parse_index, get_pair_info


class FingerprintDataset(Data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, csv_file, img_width, img_height, pad_width=0, RBS=False, PI=False):
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
        self.PI = PI
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
        image = TF.to_tensor(np.array(image))
        label = TF.to_tensor(np.array(label))

        # image = TF.normalize(image, mean=(0,), std=(1,))
        # label = TF.normalize(label, mean=(0,), std=(1,))
        return image, label

    def __getitem__(self, idx, trans=True):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = util.read_bin(self.landmarks_frame.iloc[idx, 0], (self.width, self.height), self.low_endian)
        ipp = util.read_8bit_bin(self.landmarks_frame.iloc[idx, 2], (self.width, self.height))

        if self.PI:
            util.mss_interpolation(image.astype('float32'), self.width, self.height)

        diff = image

        # # normalize
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
        # bk = util.read_bin(self.landmarks_frame.iloc[idx, 1], (self.width, self.height), self.low_endian)
        ipp_path = self.landmarks_frame.iloc[idx, 2]
        # diff = util.subtract(image, bk)
        if self.PI:
            util.mss_interpolation(image.astype('float32'), self.width, self.height)
        diff = image

        # # normalize
        diff = ((diff - np.mean(diff)) / (np.max(diff) - np.min(diff))).astype('float32')

        diff = np.pad(diff, self.pad_width, 'reflect')

        return diff, ipp_path

    def save_img(self, img, output_path):
        pad_h = self.pad_width[0][0]
        pad_w = self.pad_width[1][0]
        # crop
        img = img[pad_h:self.height, pad_w:self.width]

        # to uint8
        img = ((img - np.min(img)) * 255 / (np.max(img) - np.min(img))).astype('uint8')

        # save
        im = Image.fromarray(img)
        out_dir = os.path.dirname(output_path)
        if os.path.exists(out_dir) is False:
            try:
                os.makedirs(out_dir)
            except OSError:
                print("Creation of the directory %s failed" % out_dir)
        im.save(output_path)


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
        self.thread = 8

        gen_data0 = parse_genuines(gen_file)
        index_data0 = parse_index(index_file)
        root_dir = os.path.dirname(index_file)
        get_pair_info(gen_data0, index_data0, root_dir, csv_file)

        self.landmarks_frame = pd.read_csv(csv_file)
        self.size = self.landmarks_frame.shape[0]
        print("Get {} pairs of image".format(self.size))
        # print('perf_score = {}'.format(self.get_perf_score()))

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
        image = TF.to_tensor(np.array(image))
        label = TF.to_tensor(np.array(label))

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

        # normalize
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

    def get_img(self, img_path):
        if self.RBS:
            img_path = img_path.replace("image_bin", "image_raw")
        else:
            img_path = img_path.replace("_Img8b_", "_Img16b_")
        img_path = img_path.replace(".png", ".bin")

        img = util.read_bin(img_path, (self.width, self.height), self.low_endian)

        if self.PI:
            util.mss_interpolation(img.astype('float32'), self.width, self.height)

        img = ((img - np.min(img)) / (np.max(img) - np.min(img))).astype('float32')

        # padding
        img = np.pad(img, self.pad_width, 'reflect')

        return img

    def save_img(self, img, output_path):
        pad_h = self.pad_width[0][0]
        pad_w = self.pad_width[1][0]
        # crop
        img = img[pad_h:self.height, pad_w:self.width]

        # to uint8
        img = ((img - np.min(img)) * 255 / (np.max(img) - np.min(img))).astype('uint8')

        # save
        im = Image.fromarray(img)
        out_dir = os.path.dirname(output_path)
        if os.path.exists(out_dir) is False:
            try:
                os.makedirs(out_dir)
            except OSError:
                print("Creation of the directory %s failed" % out_dir)
            # else:
            #     print("Successfully created the directory %s " % out_dir)
        im.save(output_path)

    def get_perf_score(self):
        def thread_job(data, q):
            match_score = util.apply_perf_BinPath(data[0], data[1])
            q.put(match_score)

        def multithread(data):
            q = Queue()
            all_thread = []
            score = 0
            for i in range(len(data)):
                thread = threading.Thread(target=thread_job, args=(data[i], q))
                thread.start()
                all_thread.append(thread)
            for t in all_thread:
                t.join()
            for _ in range(len(all_thread)):
                score += q.get()
            return score

        perf_score = 0

        with progressbar.ProgressBar(max_value=self.size) as bar:
            for idx in range(0, self.size, self.thread):
                thread_data = []
                thread = self.thread
                if self.size - idx < self.thread:
                    thread = self.size - idx
                for t in range(thread):
                    thread_data.append(
                        [self.landmarks_frame.iloc[idx + t, 7], self.landmarks_frame.iloc[idx + t, 8], 0])
                score = multithread(thread_data)

                perf_score += score
                bar.update(idx)
        return perf_score


if __name__ == '__main__':
    pass
