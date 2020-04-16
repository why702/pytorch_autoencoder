from skimage.feature import local_binary_pattern
import numpy as np
import struct
import cv2
import math
import os
import csv

def read_bin(bin_path, tuple_size=(200, 200), low_endian=True):
    f = open(bin_path, "r")
    byte = np.fromfile(f, dtype=np.uint16)

    if low_endian == False:
        for i in range(len(byte)):
            b = struct.pack('>H', byte[i])
            byte[i] = struct.unpack('H', b)[0]
    return byte.reshape(tuple_size)

def read_8bit_bin(bin_path, tuple_size=(200, 200), low_endian=True):
    f = open(bin_path, "r")
    byte = np.fromfile(f, dtype=np.uint8)

    if low_endian == False:
        for i in range(len(byte)):
            b = struct.pack('>H', byte[i])
            byte[i] = struct.unpack('H', b)[0]
    return byte.reshape(tuple_size)


def read_bin_flatten(bin_path, low_endian=True):
    f = open(bin_path, "r")
    byte = np.fromfile(f, dtype=np.uint16)

    if low_endian == False:
        for i in range(len(byte)):
            b = struct.pack('>H', byte[i])
            byte[i] = struct.unpack('H', b)[0]
    return byte


def convert_lbp(byte):
    radius = 1
    n_points = 15 * radius
    threshold = 300
    byte[byte < threshold] = 0
    lbp = local_binary_pattern(byte, n_points, radius)
    return lbp


def subtract(nd1, nd2):
    diff = np.subtract(nd1.astype(np.float32), nd2.astype(np.float32))
    min = np.min(diff)
    diff -= min
    return diff


def normalize_ndarray(nd):
    return (nd - np.min(nd)) / (np.max(nd) - np.min(nd))


def normalize_ndarray_set(nd, min, max):
    return (nd - min) / (max - min)


def get_circle_boundary(byte):
    norm1 = normalize_ndarray(byte) * 255
    norm1 = norm1.astype(np.uint8)
    _, norm1_c = cv2.threshold(norm1, 10, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(norm1_c, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)

    area_list = []
    for cnt in contours:
        area_list.append(cv2.contourArea(cnt))

    area_list = np.asarray(area_list)
    max_index = np.argmax(area_list)
    cnt = contours[max_index]
    (c_x, c_y), radius = cv2.minEnclosingCircle(cnt)
    return c_x, c_y, radius


def LPF_FWHM(byte, LPF):
    L = 0.5

    # # get center
    # c_x, c_y, radius = get_circle_boundary(byte)
    # c_x = 120
    # shift_y = (byte.shape[0] / 2 - c_x) * 2 * L / byte.shape[0]
    # shift_x = (byte.shape[1] / 2 - c_y) * 2 * L / byte.shape[1]
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

    # RHO_ = normalize_ndarray(RHO) * 255
    # cv2.imshow('', RHO_.astype(np.uint8))
    # cv2.waitKey()

    # Apply localization kernel to the original image to reduce noise
    Image_orig_f = ((np.fft.fft2(byte)))
    expo = np.fft.fftshift(
        np.exp(-np.power((np.divide(RHO, math.sqrt((LPF**2) /
                                                   np.log(2)))), 2)))
    # expo = normalize_ndarray(expo) * 255
    # cv2.imshow('', expo.astype(np.uint8))
    # cv2.waitKey()

    Image_orig_filtered = np.real(
        np.fft.ifft2((np.multiply(Image_orig_f, expo))))
    return Image_orig_filtered

def read_bins(bin_dir, width, height, low_endian, FORMAT = 0):
    img_list = []
    bk_list = []
    ipp_list = []
    bds_list = []
    img = None
    bk = None
    ipp = None
    bds = None
    BK_et = 0
    need_bk = True

    for root, dirs, files in os.walk(bin_dir, topdown=False):
        for name in files:
            # print(os.path.join(root, name))

            if os.path.splitext(name)[1] == '.bin':

                if FORMAT == 702:
                    # if root.find("image_raw") != -1:
                    #     img = util.read_bin(os.path.join(root, name),
                    #                         (height, width), low_endian)
                    #     img = np.pad(img, ((2, 2), (1, 1)), 'reflect')
                    # if img is None:
                    #     continue
                    #
                    # # diff = util.normalize_ndarray(img) * 255
                    # # cv2.imshow("", diff.astype(np.uint8))
                    # # cv2.waitKey()
                    #
                    # self.input_list.append(img)
                    pass
                else:

                    if name.find("_Img16b_") != -1:

                        #find et
                        et = name[name.find("_et=") + 4: name.find("_hc=")]

                        #find bk
                        mi = name[name.find("mica=") + 5: name.find("mica=") + 7]

                        if root.find("enroll") != -1 and mi == "00" and need_bk:
                            bk_name = name.replace("Img16b", "Img16bBkg")
                            bk = read_bin(os.path.join(root, bk_name))
                            need_bk = False
                            BK_et = et
                        elif need_bk == False and root.find("enroll") == -1:
                            need_bk = True

                        if BK_et != et or et == 0:
                            continue

                        img = read_bin(os.path.join(root, name))

                        ipp_name = name.replace("Img16b", "Img8b")
                        ipp = read_8bit_bin(os.path.join(root, ipp_name))

                        bds_name = name.replace("Img16b", "Img16bBkg")
                        bds = read_bin(os.path.join(root, bds_name))

                        if img is None or bk is None or ipp is None or bds is None:
                            continue

                        img_list.append(img)
                        bk_list.append(bk)
                        ipp_list.append(ipp)
                        bds_list.append(bds)

    print("img_list size is {}".format(len(img_list)))
    return img_list, bk_list, ipp_list, bds_list

def read_bins_toCSV(bin_dir, out_path, width, height, low_endian, FORMAT = 0, GOOD = False):
    BK_et = 0
    need_bk = True
    count = 0

    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for root, dirs, files in os.walk(bin_dir, topdown=False):
            for name in files:
                # print(os.path.join(root, name))

                if os.path.splitext(name)[1] == '.bin':

                    if FORMAT == 702:
                        # if root.find("image_raw") != -1:
                        #     img = util.read_bin(os.path.join(root, name),
                        #                         (height, width), low_endian)
                        #     img = np.pad(img, ((2, 2), (1, 1)), 'reflect')
                        # if img is None:
                        #     continue
                        #
                        # # diff = util.normalize_ndarray(img) * 255
                        # # cv2.imshow("", diff.astype(np.uint8))
                        # # cv2.waitKey()
                        #
                        # self.input_list.append(img)
                        pass
                    else:

                        if name.find("_Img16b_") != -1:

                            #find et
                            et = name[name.find("_et=") + 4: name.find("_hc=")]

                            #find mica
                            mi = name[name.find("mica=") + 5: name.find("mica=") + 7]

                            #find egp
                            egp = int(name[name.find("_egp=") + 5: name.find("_rl=")])

                            #find rl
                            rl = int(name[name.find("_rl=") + 4: name.find("_CxCy")])

                            if GOOD and egp < 80 or rl > 0:
                                continue

                            if root.find("enroll") != -1 and mi == "00" and need_bk:
                                bk_name = name.replace("Img16b", "Img16bBkg")
                                bk = os.path.join(root, bk_name)
                                need_bk = False
                                BK_et = et
                            elif need_bk == False and root.find("enroll") == -1:
                                need_bk = True

                            if BK_et != et or et == 0:
                                continue

                            img = os.path.join(root, name)

                            ipp_name = name.replace("Img16b", "Img8b")
                            ipp = os.path.join(root, ipp_name)

                            bds_name = name.replace("Img16b", "Img16bBkg")
                            bds = os.path.join(root, bds_name)

                            if os.path.exists(img) is False or os.path.exists(bk) is False or os.path.exists(ipp) is False or os.path.exists(bds) is False:
                                continue

                            writer.writerow([img, bk, ipp, bds])
                            count += 1

    print("img_list size is {}".format(count))
    return


if __name__ == '__main__':
    pass
