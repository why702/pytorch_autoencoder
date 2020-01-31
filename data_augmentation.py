import numpy as np
from scipy import misc
# from PIL import Image
import cv2
import os
import util
import skimage
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops

DEBUG = False

FORMAT = 702

class configuration():
    def __init__(self,
                 bin_dir,
                 ratio=0.9,
                 image_size=(200, 200),
                 batch_size=32,
                 epoch_size=20):
        self.image_size = image_size
        self.Width = self.image_size[1]
        self.Height = self.image_size[0]
        self.bin_dir = bin_dir
        self.read_bins()

        self.batch_size = batch_size
        # self.epoch_size = epoch_size  # Number of batches per epoch.
        n_train = int(len(self.input_list) * ratio)
        self.train_list = self.input_list[:n_train]
        self.val_list = self.input_list[n_train:]

        self.epoch_size = int(len(self.train_list) / batch_size)

        # 1: Random rotate 2: Random crop  4: Random flip  8:  Fixed image standardization  16: Flip
        self.RANDOM_ROTATE = 1
        self.RANDOM_CROP = 2
        self.RANDOM_FLIP = 4
        self.FIXED_STANDARDIZATION = 8
        self.FLIP = 16

    def run(self, input_placeholder, control_placeholder,
            batch_size_placeholder):

        index_queue = tf.train.range_input_producer(len(self.train_list))

        index_dequeue_op = index_queue.dequeue_many(
            self.batch_size * self.epoch_size, 'index_dequeue')

        input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                              dtypes=[tf.float32, tf.int32],
                                              shapes=[(
                                                  self.Height,
                                                  self.Width,
                                                  1,
                                              ), (1, )],
                                              shared_name=None,
                                              name=None)
        enqueue_op = input_queue.enqueue_many(
            [input_placeholder, control_placeholder], name='enqueue_op')

        nrof_preprocess_threads = 1
        image_batch = self.create_input_pipeline(input_queue,
                                                 (self.Height, self.Width),
                                                 nrof_preprocess_threads,
                                                 batch_size_placeholder)
        return image_batch, enqueue_op, index_dequeue_op

    def read_bins(self):
        self.input_list = []

        for root, dirs, files in os.walk(self.bin_dir, topdown=False):
            for name in files:
                # print(os.path.join(root, name))

                if os.path.splitext(name)[1] == '.bin':

                    if FORMAT == 702:
                        if root.find("image_raw") != -1:
                            img = util.read_bin(os.path.join(root, name),
                                                (188, 134), False)
                            img = np.pad(img, ((2, 2), (1, 1)), 'reflect')
                        if img is None:
                            continue

                        # diff = util.normalize_ndarray(img) * 255
                        # cv2.imshow("", diff.astype(np.uint8))
                        # cv2.waitKey()

                        self.input_list.append(img)
                    else:
                        img = None
                        bkg = None
                        if name.find("16bitPreImage") != -1:
                            bkg_name = name.replace("16bitPreImage",
                                                    "16bitBkg")
                            img = util.read_bin(os.path.join(root, name))
                            bkg = util.read_bin(os.path.join(root, bkg_name))
                        elif name.find("Image16bit") != -1:
                            bkg_name = name.replace("16bitPreImage",
                                                    "Image16bitBkg")
                            img = util.read_bin(os.path.join(root, name))
                            bkg = util.read_bin(os.path.join(root, bkg_name))
                        elif name.find("Img16b") != -1:
                            bkg_name = name.replace("Img16b", "Img16bBkg")
                            img = util.read_bin(os.path.join(root, name))
                            bkg = util.read_bin(os.path.join(root, bkg_name))

                        if img is None or bkg is None:
                            continue

                        diff = util.substract(img, bkg)
                        # diff = util.normalize_ndarray(diff) * 255
                        # diff = diff.astype(np.uint8)

                        # diff = LPF_FWHM(diff)
                        # byte = read_bin_flatten(os.path.join(bin_dir, file)).astype(np.float32)

                        # diff = util.normalize_ndarray(diff) * 255
                        # cv2.imshow("", diff.astype(np.uint8))
                        # cv2.waitKey()

                        self.input_list.append(diff)

        print("input_list size is {}".format(len(self.input_list)))
        return np.random.shuffle(self.input_list)

    def create_input_pipeline(self, input_queue, image_size,
                              nrof_preprocess_threads, batch_size_placeholder):
        images_list = []
        for _ in range(nrof_preprocess_threads):
            # get the filenames, control from the Queue
            image, control = input_queue.dequeue()
            # image = tf.cond(
            #     get_control_flag(control[0],
            #                      self.RANDOM_ROTATE), lambda: tf.py_func(
            #                          random_rotate_image, [image], tf.float32),
            #     lambda: tf.identity(image))
            image = tf.cond(
                get_control_flag(control[0],
                                 self.RANDOM_CROP), lambda: tf.random_crop(
                                     image, image_size + (1, )),
                lambda: tf.image.resize_image_with_crop_or_pad(
                    image, image_size[0], image_size[1]))
            image = tf.cond(get_control_flag(
                control[0],
                self.RANDOM_FLIP), lambda: tf.image.random_flip_left_right(
                    image), lambda: tf.identity(image))
            image = tf.cond(
                get_control_flag(control[0],
                                 self.FIXED_STANDARDIZATION), lambda: (tf.cast(
                                     image, tf.float32) - 127.5) / 128.0,
                lambda: tf.image.per_image_standardization(image))
            image = tf.cond(get_control_flag(control[0],
                                             self.FLIP), lambda: tf.image.
                            flip_left_right(image), lambda: tf.identity(image))

            image.set_shape(image_size + (1, ))
            images_list.append([[image]])

        image_batch = tf.train.batch_join(images_list,
                                          batch_size=batch_size_placeholder,
                                          shapes=[image_size + (1, )],
                                          enqueue_many=True,
                                          capacity=4 *
                                          nrof_preprocess_threads * 100,
                                          allow_smaller_final_batch=True)

        return image_batch


def random_rotate_image(image):
    angle = np.random.uniform(low=-180.0, high=180.0)
    # rows, cols = image.shape
    # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    # dst = cv2.warpAffine(image, M, (cols, rows))
    # return dst
    return skimage.transform.rotate(image, angle)


def get_control_flag(control, field):
    return tf.equal(tf.mod(tf.floor_div(control, field), 2), 1)


def LPF_FWHM(byte, LPF=0.13):
    L = 0.5
    shift_x = shift_y = 0

    x = np.linspace(-L + shift_x, L + shift_x, byte.shape[0])
    y = np.linspace(-L + shift_y, L + shift_y, byte.shape[1])
    [X1, Y1] = (np.meshgrid(x, y))
    X = X1.T
    Y = Y1.T

    def cart2pol(x, y):
        theta = np.arctan2(y, x)
        rho = np.hypot(x, y)
        return (theta, rho)

    [THETA, RHO] = cart2pol(X, Y)

    # Apply localization kernel to the original image to reduce noise
    Image_orig_f = ((np.fft.fft2(byte)))
    expo = np.fft.fftshift(
        np.exp(-np.power((np.divide(RHO, math.sqrt((LPF**2) /
                                                   np.log(2)))), 2)))

    Image_orig_filtered = np.real(
        np.fft.ifft2((np.multiply(Image_orig_f, expo))))
    return Image_orig_filtered


if __name__ == '__main__':
    img_dir = '/home/bill/svn/Improving_Unsupervised_Defect_Segmentation/img'
    name_list = []
    for file in os.listdir(img_dir):
        if file.endswith('.jpg'):
            name_list.append(os.path.join(img_dir, file))

    conf = configuration(5, name_list)
    batch_x = conf.train_batch()
