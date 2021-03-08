from skimage.feature import local_binary_pattern
import numpy as np
import struct
import cv2
import math
import os
import csv
import re
import shutil
import subprocess


def read_bin(bin_path, tuple_size=(200, 200), low_endian=True):
    f = open(bin_path, "r")
    byte = np.fromfile(f, dtype=np.uint16)

    if low_endian == False:
        for i in range(len(byte)):
            b = struct.pack('>H', byte[i])
            byte[i] = struct.unpack('H', b)[0]
    return byte.reshape(tuple_size)


def read_8bit_bin(bin_path, tuple_size=(200, 200)):
    if os.path.splitext(bin_path)[1] == '.png':
        img = cv2.imread(bin_path, 0)
        return img
    else:
        f = open(bin_path, "r")
        byte = np.fromfile(f, dtype=np.uint8)

        # if low_endian == False:
        #     for i in range(len(byte)):
        #         b = struct.pack('>H', byte[i])
        #         byte[i] = struct.unpack('H', b)[0]
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
        np.exp(-np.power((np.divide(RHO, math.sqrt((LPF ** 2) /
                                                   np.log(2)))), 2)))
    # expo = normalize_ndarray(expo) * 255
    # cv2.imshow('', expo.astype(np.uint8))
    # cv2.waitKey()

    Image_orig_filtered = np.real(
        np.fft.ifft2((np.multiply(Image_orig_f, expo))))
    return Image_orig_filtered


def find_bk_bins(bin_dir):
    pair_EtBk = []
    for root, dirs, files in os.walk(bin_dir, topdown=False):
        for name in files:
            pos = name.lower().find('_bkg.bin')
            if pos >= 0:
                et = int(name[0:pos])
                pair_EtBk.append((et, os.path.join(root, name)))
    return pair_EtBk


def read_bins(bin_dir, width, height, RBS=False):
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
    low_endian = not RBS

    for root, dirs, files in os.walk(bin_dir, topdown=False):
        for name in files:
            if os.path.splitext(name)[1] == '.bin':
                if RBS:
                    if root.find("image_raw") != -1:
                        img = read_bin(os.path.join(root, name), low_endian)

                        ipp_name = name.replace("image_raw", "image_bin")
                        ipp = read_8bit_bin(os.path.join(root, ipp_name))

                        bds_name = name.replace("image_raw", "image_bkg")
                        bds = read_bin(os.path.join(root, bds_name), low_endian)

                        # if img is None or bk is None or ipp is None or bds is None:
                        #     continue
                        if img is None or ipp is None or bds is None:
                            continue

                        img_list.append(img)
                        bk_list.append(bk)
                        ipp_list.append(ipp)
                        bds_list.append(bds)
                    pass
                else:

                    if name.find("_Img16b_") != -1:

                        # find et
                        et = name[name.find("_et=") + 4: name.find("_hc=")]

                        # find bk
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

                        img = read_bin(os.path.join(root, name), low_endian)

                        ipp_name = name.replace("Img16b", "Img8b")
                        ipp = read_8bit_bin(os.path.join(root, ipp_name))

                        bds_name = name.replace("Img16b", "Img16bBkg")
                        bds = read_bin(os.path.join(root, bds_name), low_endian)

                        if img is None or bk is None or ipp is None or bds is None:
                            continue

                        img_list.append(img)
                        bk_list.append(bk)
                        ipp_list.append(ipp)
                        bds_list.append(bds)

    print("img_list size is {}".format(len(img_list)))
    return img_list, bk_list, ipp_list, bds_list


def read_bins_toCSV(bin_dir, out_path, width, height, RBS=False, GOOD=False):
    BK_et = 0
    need_bk = True
    count = 0
    img = None
    bk = None
    ipp = None
    bds = None
    pair_EtBk = find_bk_bins(bin_dir)

    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for root, dirs, files in os.walk(bin_dir, topdown=False):
            for name in files:
                # print(os.path.join(root, name))

                if os.path.splitext(name)[1] == '.bin':

                    if RBS:
                        if root.find("image_raw") != -1:
                            img = os.path.join(root, name)

                            ipp_root = root.replace("image_raw", "image_bin")
                            ipp_name = name.replace("bin", "png")
                            ipp = os.path.join(ipp_root, ipp_name)

                            bds_root = root.replace("image_raw", "image_bkg")
                            bds = os.path.join(bds_root, name)

                            bk = None
                            pos = name.lower().find('_et=')
                            if pair_EtBk is not [] and pos >= 0:
                                sEt = name.lower()[pos + 4:]
                                pos_end = sEt.find('_')
                                iEt = int(float(sEt[0:pos_end]) * 1000)
                                for et_, bk_ in pair_EtBk:
                                    if et_ == iEt:
                                        bk = bk_
                            if bk is None:
                                bk = img

                            if os.path.exists(img) is False or os.path.exists(
                                    ipp) is False or os.path.exists(bds) is False:
                                continue

                            writer.writerow([img, bk, ipp, bds])
                            count += 1
                    else:

                        if name.find("_Img16b_") != -1:

                            # find et
                            et = name[name.find("_et=") + 4: name.find("_hc=")]

                            # find mica
                            mi = name[name.find("mica=") + 5: name.find("mica=") + 7]

                            # find egp rl
                            egp = 100
                            rl = 0
                            if name.find("_egp=") >= 0 and name.find("_rl=") >= 0 and name.find("_CxCy=") >= 0:
                                egp = int(name[name.find("_egp=") + 5: name.find("_rl=")])
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

                            if os.path.exists(img) is False or os.path.exists(bk) is False or os.path.exists(
                                    ipp) is False or os.path.exists(bds) is False:
                                continue

                            writer.writerow([img, bk, ipp, bds])
                            count += 1

    print("img_list size is {}".format(count))
    return


def parse_genuines(gen_file):
    with open(gen_file, 'r') as file:
        all_lines = file.readlines()

    data = []
    for line in all_lines:
        if line[0:1] == "#":
            continue
        match = re.findall(r"\s*([0-9]+)\s*", line)
        if len(match) >= 23:
            info = dict()
            info['enroll'] = match[0] + match[1] + match[2]
            info['verify'] = match[3] + match[4] + match[5]
            info['match'] = match[6]
            info['score'] = match[11]
            data.append(info)
    return data


def run_perf_sum_score(test_folder, org=False):
    # write index file
    write_fpdboncex_cmd = "python ..\\read_sys_file\\generate_4folder.py {} > {}\\i.fpdbindex".format(test_folder,
                                                                                                      test_folder)
    os.system(write_fpdboncex_cmd)

    # execute perf
    key = "tst"
    if org:
        key = "org"
    output_perf = ".\\test\\{}".format(key)
    if os.path.exists(output_perf) and org is False:
        shutil.rmtree(output_perf)
    perf_cmd = "..\\PerfEval_win_64.exe -skip -rs={} -n=test -db_mask -Aeval.inverted_mask=1 -improve_dry=94 -latency_adjustment=0 -algo=egistec_200x200_cardo_3PG_CH1JSC_H -tp=image -api=mobile -ver_type=dec -far=1:100K -ms=allx:ogi -enr=1000of15+g -div=1000 -Cmaxtsize=1024000 -ver_update=gen -scorefiles=1 -static_pattern_detect -threads=1 -Agen.aperture.radius=120 -Agen.aperture.x=107 -Agen.aperture.y=87  \"{}\\i.fpdbindex\" > perf_info.txt".format(
        key, test_folder)
    os.system(perf_cmd)

    # read genuines.txt
    genuines_path = output_perf + "\\genuines.txt"
    genuines_info = parse_genuines(genuines_path)
    sum_score = 0
    score_array = []
    for info in genuines_info:
        sum_score += int(info['score'])
        score_array.append(int(info['score']))
    return sum_score, score_array


def runcmd(command):
    try:
        ret = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
                             timeout=5)
        if ret.returncode == 0:
            # print("success:", ret)
            return ret.stdout
        else:
            print("error:", ret)
            return False
    except subprocess.CalledProcessError as e:
        print(e.output)
        return False


def apply_perf(raw_e, raw_v):
    perf_result = []
    perf_score = 0
    for i in range(raw_e.shape[0]):
        bin_e = raw_e[i].astype('uint8')
        bin_v = raw_v[i].astype('uint8')
        f_e = open('e.bin', 'w+b')
        binary_format = bytearray(bin_e)
        f_e.write(binary_format)
        f_e.close()
        f_v = open('v.bin', 'w+b')
        binary_format = bytearray(bin_v)
        f_v.write(binary_format)
        f_v.close()
        stdout = runcmd('PBexe.exe e.bin v.bin')  # match_score = 57133, rot = 0, dx = 0, dy = 0,
        match_score = -1
        if str(stdout):
            pos_str = stdout.find('match_score = ') + 14
            pos_end = stdout.find(', rot', pos_str)
            match_score = int(stdout[pos_str: pos_end])
        perf_result.append(match_score)
        perf_score += match_score
    print('perf_score = {}'.format(perf_score))
    return perf_result


def show_ndarray(img, name):
    img = np.float32(img)
    norm = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    norm = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imshow(name, norm)
    cv2.waitKey(1000)


def pattern_interpolation(img, strx, stry, endx, endy):
    for i in range(stry, endy, 30):
        for j in range(strx, endx, 30):
            for k in range(0, 9, 4):
                for l in range(0, 8, 7):
                    pos_x0 = j + k
                    pos_y0 = i + l
                    pos_x1 = j + k + 1
                    pos_y1 = i + l + 1
                    pos_x2 = j + k + 0
                    pos_y2 = i + l + 2
                    sum0 = (img[pos_y0 - 1, pos_x0 - 1] +
                            img[pos_y0 + 0, pos_x0 - 1] +
                            img[pos_y0 + 1, pos_x0 - 1] +
                            img[pos_y0 - 1, pos_x0 + 0] +
                            img[pos_y0 + 1, pos_x0 + 0] +
                            img[pos_y0 - 1, pos_x0 + 1] +
                            img[pos_y0 + 0, pos_x0 + 1]) / 7
                    sum1 = (img[pos_y1 + 0, pos_x1 - 1] +
                            img[pos_y1 - 1, pos_x1 + 0] +
                            img[pos_y1 + 1, pos_x1 + 0] +
                            img[pos_y1 - 1, pos_x1 + 1] +
                            img[pos_y1 + 0, pos_x1 + 1] +
                            img[pos_y1 + 1, pos_x1 + 1]) / 6
                    sum2 = (img[pos_y2 - 1, pos_x2 - 1] +
                            img[pos_y2 + 0, pos_x2 - 1] +
                            img[pos_y2 + 1, pos_x2 - 1] +
                            img[pos_y2 - 1, pos_x2 + 0] +
                            img[pos_y2 + 1, pos_x2 + 0] +
                            img[pos_y2 + 0, pos_x2 + 1] +
                            img[pos_y2 + 1, pos_x2 + 1]) / 7
                    img[pos_x0, pos_y0] = sum0
                    img[pos_x1, pos_y1] = sum1
                    img[pos_x2, pos_y2] = sum2


def mss_interpolation(img, width, height):
    pattern_interpolation(img, 16, 16, width - 20, height - 20)
    pattern_interpolation(img, 31, 31, width, height)
    # #721
    # pattern_interpolation(img, 12, 12, width - 20, height - 20)
    # pattern_interpolation(img, 27, 27, width - 20, height - 20)
    # pattern_interpolation(img, 12, 162, width - 20, 13)


if __name__ == '__main__':
    pass
