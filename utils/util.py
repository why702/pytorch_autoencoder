import csv
import math
import os
import re
import shutil
import struct
import subprocess
import threading
from queue import Queue

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


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


# egistec_200x200_cardo_2PX
# egistec_200x200_cardo_2PA
# egistec_200x200_cardo_2PA_NEW
# egistec_200x200_cardo_2PB
# egistec_200x200_cardo_2PB_CH1M30
# egistec_200x200_cardo_3PX
# egistec_200x200_cardo_3PC
# egistec_200x200_cardo_3PD
# egistec_200x200_cardo_3PDx
# egistec_193x193_cardo_3PA
# egistec_193x193_cardo_3PF
# egistec_120x33_cardo_525
# egistec_120x27_cardo_5XX
# egistec_120x25_cardo_528
# egistec_134x188_cardo_702_NEW
# egistec_134x188_cardo_702_INV
# egistec_134x188_cardo_702
# egistec_200x200_cardo_CH1ABY
# egistec_200x200_cardo_CH1AJA
# egistec_200x200_cardo_CH1AJB
# egistec_200x200_cardo_CH1AJA_demorie
# egistec_200x200_cardo_CH1AJB_demorie
# egistec_200x200_cardo_CH1LA
# egistec_200x200_cardo_CH1LA_NEW
# egistec_200x200_cardo_ET760
# egistec_142x142_cardo_ET760_CROP
# egistec_150x104_cardo_ET760_CROP2
# egistec_134x188_cardo_CH1M30
# egistec_134x188_cardo_CH1M30_INV
# egistec_132x120_cardo_CH1M30
# egistec_200x200_cardo_CH1E_SB
# egistec_200x200_cardo_CH1E_SV
# egistec_200x200_cardo_CH1E_H
# egistec_200x200_cardo_CH1B_H
# egistec_200x200_cardo_CH1J_SB
# egistec_200x200_cardo_CL1MH2
# egistec_134x188_cardo_CL1MH2
# egistec_134x188_cardo_CL1MH2_INV
# egistec_118x172_cardo_CL1WING
# egistec_134x188_cardo_CL1WING
# egistec_134x188_cardo_CL1WING_Latency
# egistec_200x200_cardo_CL1MH2_CLT3
# egistec_134x188_cardo_CL1MH2_C230
# egistec_200x200_cardo_CL1MH2V
# egistec_193x193_cardo_CL1TIME
# egistec_193x193_cardo_CL1CAY
# egistec_193x193_cardo_CL1CAY_pad
# egistec_200x200_cardo_CO1D151
# egistec_200x200_cardo_CO1A118
# egistec_200x200_cardo_3PG_CO1A118
# egistec_200x200_cardo_3PG_CH1JSC_H
# egistec_200x200_cardo_CS3ZE2
# egistec_200x200_cardo_CV1CPD1960
# egistec_200x200_cardo_CV1CTD2041
# egistec_200x200_cardo_3PG_CV1CPD1960
# egistec_150x150_cardo_ET901
# egistec_150x150_cardo_ET901_CL1V60
# egistec_175x175_cardo_EF9002
# egistec_175x175_cardo_EF9002_raw
# egistec_200x200_cardo_S3PG1
# egistec_200x200_cardo_S3PG2
# egistec_200x200_cardo_S3PG3
# egistec_200x200_cardo_S3PG3_Latency
# egistec_200x200_cardo_S3PG4
# egistec_200x200_cardo_S3PG5
# egistec_200x200_cardo_S3PG6
# egistec_200x200_cardo_S3PG6_new
# egistec_200x200_evo_S3PG6
# egistec_200x200_cardo_S3PG7
# egistec_200x200_cardo_S3PG7_new
# egistec_200x200_evo_S3PG7
# egistec_200x200_cardo_S3PG8
# egistec_200x200_cardo_S3PG8_new
# egistec_200x200_cardo_S2PB1
# egistec_200x200_cardo_S2PA4
# egistec_193x193_cardo_S3PF5
# egistec_193x193_cardo_S3PF2
# egistec_193x193_cardo_S3PA2
# egistec_193x193_cardo_S3PA2_Latency
# egistec_134x188_cardo_SXC210
# egistec_193x193_cardo_ET715_3PG
# egistec_215x175_cardo_EL721_3PI_CV1CTD2052
# egistec_215x175_evo_EL721_3PI_CV1CTD2052
# egistec_200x200_cardo_ET760_2
# egistec_200x200_cardo_ET760_2_IPP61e
# egistec_200x200_cardo_ET760_3
# egistec_215x175_cardo_EL721_3PI_S3PI1
# egistec_215x175_evo_EL721_3PI_S3PI1
# egistec_200x200_cardo_CH2NTH_B
# egistec_200x200_cardo_CH2NTH_V
# egistec_200x200_cardo_CH2NTH
# gen_0x0_eval_cardo
# gen_130x130_neo
# gen_130x130_neo_speed
# gen_192x192_spectral
# gen_192x192_minutiae
# gen_192x192_minutiae_speed_mem
# gen_80x64_cardo_capacitive
# gen_6x6_cardo_embedded_363dpi
# gen_6x6_cardo_embedded_508dpi
# gen_10x10_hybrid_embedded_254dpi
# gen_10x10_cardo_embedded_363dpi
# gen_8x8_hybrid_embedded_254dpi
# gen_8x8_hybrid_plus_embedded_254dpi
# gen_8x8_cardo_embedded_363dpi
# gen_fullsize_cardo_embedded_254dpi
def run_perf_sum_score(test_folder, org=False):
    # write index file
    write_fpdboncex_cmd = "python ..\\..\\read_sys_file\\generate_4folder.py {} > {}\\i.fpdbindex".format(test_folder,
                                                                                                          test_folder)
    os.system(write_fpdboncex_cmd)

    # execute perf
    key = "tst"
    if org:
        key = "org"
    output_perf = ".\\test\\{}".format(key)
    if os.path.exists(output_perf) and org is False:
        shutil.rmtree(output_perf, ignore_errors=True)
    perf_cmd = "..\\..\\PerfEval_win_64.exe -skip -rs={} -n=test -db_mask -Aeval.inverted_mask=1 -improve_dry=94 -latency_adjustment=0 -algo=egistec_200x200_cardo_CH1AJA -tp=image -api=mobile -ver_type=dec -far=1:100K -ms=allx:ogi -enr=1000of15+g -div=1000 -Cmaxtsize=1024000 -ver_update=gen -scorefiles=1 -static_pattern_detect -threads=4 -Agen.aperture.radius=120 -Agen.aperture.x=107 -Agen.aperture.y=87  \"{}\\i.fpdbindex\" > perf_info.txt".format(
        key, test_folder)
    print('run\n{}'.format(perf_cmd))
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
        PBexe_path = os.path.join(os.path.dirname(__file__), 'PBexe.exe')
        stdout = runcmd('{} e.bin v.bin'.format(PBexe_path))  # match_score = 57133, rot = 0, dx = 0, dy = 0,
        match_score = -1
        if stdout:
            pos_str = stdout.find('match_score = ') + 14
            pos_end = stdout.find(', rot', pos_str)
            match_score = int(stdout[pos_str: pos_end])
        perf_result.append(match_score)
        perf_score += match_score
    print('perf_score = {}'.format(perf_score))
    return perf_result

def apply_perf_thread(raw_e, raw_v):
    def thread_job(data):
        data[2] = apply_perf_BinPath(data[0], data[1])

    def multithread(data):
        all_thread = []
        for i in range(len(data)):
            thread = threading.Thread(target=thread_job, args=data[i])
            thread.start()
            all_thread.append(thread)
        for t in all_thread:
            t.join()
        score_array = data[:,2]
        return score_array

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
        PBexe_path = os.path.join(os.path.dirname(__file__), 'PBexe.exe')
        stdout = runcmd('{} e.bin v.bin'.format(PBexe_path))  # match_score = 57133, rot = 0, dx = 0, dy = 0,
        match_score = -1
        if stdout:
            pos_str = stdout.find('match_score = ') + 14
            pos_end = stdout.find(', rot', pos_str)
            match_score = int(stdout[pos_str: pos_end])
        perf_result.append(match_score)
        perf_score += match_score
    print('perf_score = {}'.format(perf_score))
    return perf_result


def apply_perf_BinPath(bin_e, bin_v):
    PBexe_path = os.path.join(os.path.dirname(__file__), 'PBexe.exe')
    stdout = runcmd('{} {} {}'.format(PBexe_path, bin_e, bin_v))  # match_score = 57133, rot = 0, dx = 0, dy = 0,
    match_score = -1
    if stdout:
        pos_str = stdout.find('match_score = ') + 14
        pos_end = stdout.find(', rot', pos_str)
        match_score = int(stdout[pos_str: pos_end])
    return match_score


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
