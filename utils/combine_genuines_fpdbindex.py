import argparse
import csv
import os
import re
from shutil import copyfile


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
            info['rotation'] = match[7]
            info['dx'] = match[8]
            info['dy'] = match[9]
            info['score'] = match[11]
            data.append(info)
    return data


def parse_index(index_file):
    with open(index_file, 'r') as file:
        all_lines = file.readlines()

    data = []
    for line in all_lines:
        if line[0:1] == "#":
            continue
        match = re.findall(r"\s*([0-9a-zA-Z\\\-\_\=\[\]\.]+)\s*", line)
        if len(match) >= 5:
            info = dict()
            info['id'] = match[0] + match[1] + match[3]
            info['path'] = match[4]
            data.append(info)
    return data


def find_path(index_data, id):
    for info in index_data:
        if info['id'] == id:
            return info['path']


def find_gen_info(gen_data, id):
    for info in gen_data:
        if info['verify'] == id:
            return info


expr = re.compile(
    r"(\S+)\\(\d+)\\(enroll|verify|identify)\\(st|45d|90d|135d)\\([0-9]+\\)*(\S+).png"
)
expr_partial = re.compile(
    r"(\S+)\\(\d+)\\(enroll|verify)\\(\S+)\\([0-9]+)\\(\S+).png"
)
expr_simple_partial = re.compile(
    r"(\S+)\\(\d+)\\(enroll|verify)\\([0-9]+)\\(\S+).png"
)
expr_enroll = re.compile(
    r"(\S+)\\(\d+)\\(enroll)\\(\S+).png"
)


def analysis_info(gen_data0, index_data0, output_file):
    root_folder = os.path.dirname(os.path.abspath(output_file))

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['enroll', 'verify', 'match', 'score', 'path', 'person', 'finger', 'verify', 'quality', 'cond', 'part',
             'fr'])
        # writer.writerow(
        #     ['person', 'finger', '50', '75', '90', '100'])
        dicts = []

        for info0 in gen_data0:
            verify_id = info0['verify']

            # find path
            path = find_path(index_data0, verify_id)

            m = expr.match(path.lower())
            m_p = expr_partial.match(path.lower())
            m_sp = expr_simple_partial.match(path.lower())
            m_e = expr_enroll.match(path.lower())

            person = ""
            finger = -1
            verify = ""
            quality = ""
            cond = ""
            part = ""

            if (m):
                person = m.group(1)
                finger = int(m.group(2))
                verify = m.group(3)
                quality = m.group(4)
                part = m.group(5)
            elif (m_p):
                person = m_p.group(1)
                finger = int(m_p.group(2))
                verify = m_p.group(3)
                cond = m_p.group(4)
                part = m_p.group(5)
            elif (m_sp):
                person = m_sp.group(1)
                finger = int(m_sp.group(2))
                verify = m_sp.group(3)
                part = m_sp.group(4)
            elif (m_e):
                person = m_e.group(1)
                finger = int(m_e.group(2))

            # replace
            person = person.replace("\\", "")
            person = person.replace('np', "")
            person = person.replace("ipp", "")
            person = person.replace("org", "")
            person = person.replace("md1", "")
            person = person.replace("md2", "")
            person = person.replace("sdk", "")
            person = person.replace("full", "")
            person = person.replace("normal", "")
            person = person.replace("_", "")
            person = person.replace("pcopy", "")
            person = person.replace("copy", "")
            person = person.replace("md3", "")

            fr = 1
            if info0['match'] == "1":
                fr = 0

            writer.writerow(
                [info0['enroll'], info0['verify'], info0['match'], info0['score'], path, person, finger, verify,
                 quality, cond, part, fr])

            # copy image
            if path.find('_md3') >= 0 and info0['match'] == '1':
                img_path = os.path.join(root_folder, path)
                out_path = os.path.join(root_folder, "md3", path)
                if not os.path.exists(os.path.dirname(out_path)):
                    os.makedirs(os.path.dirname(out_path))
                copyfile(img_path, out_path)
            elif info0['match'] == '0':
                img_path = os.path.join(root_folder, path)
                out_path = os.path.join(root_folder, "fail", path)
                if not os.path.exists(os.path.dirname(out_path)):
                    os.makedirs(os.path.dirname(out_path))
                copyfile(img_path, out_path)

        #     if part != '100':
        #         id = str(person) + "\t" + str(finger) + "\t0" + str(part)
        #     else:
        #         id = str(person) + "\t" + str(finger) + "\t" + str(part)
        #
        #     saved = 0
        #     for d in dicts:
        #         if d['id'] == id:
        #             d['sum'] += fr
        #             d['count'] += 1
        #             saved = 1
        #
        #     if saved == 0:
        #         dicts.append({'id': id, 'person': person, 'finger': finger, 'part': part, 'sum': fr, 'count': 1})
        #
        # dicts.sort(key=lambda k: k['id'])
        #
        # d50_sum = 0
        # d50_count = 0
        # d75_sum = 0
        # d75_count = 0
        # d90_sum = 0
        # d90_count = 0
        # d100_sum = 0
        # d100_count = 0
        # for i in range(0,len(dicts),4):
        #     d50 = dicts[i]['sum'] / dicts[i]['count']
        #     d75 = dicts[i+1]['sum'] / dicts[i+1]['count']
        #     d90 = dicts[i+2]['sum'] / dicts[i+2]['count']
        #     d100 = dicts[i+3]['sum'] / dicts[i+3]['count']
        #     writer.writerow([dicts[i]['person'], dicts[i]['finger'], d50, d75, d90, d100, dicts[i]['sum'], dicts[i]['count'], dicts[i+1]['sum'], dicts[i+1]['count'], dicts[i+2]['sum'], dicts[i+2]['count'], dicts[i+3]['sum'], dicts[i+3]['count']])
        #     d50_sum += dicts[i]['sum']
        #     d50_count += dicts[i]['count']
        #     d75_sum += dicts[i + 1]['sum']
        #     d75_count += dicts[i + 1]['count']
        #     d90_sum += dicts[i + 2]['sum']
        #     d90_count += dicts[i + 2]['count']
        #     d100_sum += dicts[i + 3]['sum']
        #     d100_count += dicts[i + 3]['count']
        #
        # writer.writerow(['person','finger', d50_sum / d50_count, d75_sum / d75_count, d90_sum / d90_count, d100_sum / d100_count])
    pass


def get_pair_info(gen_data0, index_data0, root_dir, output_file, GOOD=True):
    root_folder = os.path.dirname(os.path.abspath(output_file))

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['enroll', 'verify', 'match', 'score', 'rot', 'dx', 'dy', 'enroll_path', 'verify_path', 'person', 'finger',
             'verify', 'quality', 'cond', 'part', 'fr'])
        # writer.writerow(
        #     ['person', 'finger', '50', '75', '90', '100'])
        dicts = []

        for info0 in gen_data0:
            if GOOD and int(info0['score']) < 20000:
                continue

            enroll_id = info0['enroll']
            verify_id = info0['verify']

            # find path
            enroll_path = os.path.join(root_dir, find_path(index_data0, enroll_id))
            verify_path = os.path.join(root_dir, find_path(index_data0, verify_id))

            m = expr.match(verify_path.lower())
            m_p = expr_partial.match(verify_path.lower())
            m_sp = expr_simple_partial.match(verify_path.lower())
            m_e = expr_enroll.match(verify_path.lower())

            person = ""
            finger = -1
            verify = ""
            quality = ""
            cond = ""
            part = ""

            if (m):
                person = m.group(1)
                finger = int(m.group(2))
                verify = m.group(3)
                quality = m.group(4)
                part = m.group(5)
            elif (m_p):
                person = m_p.group(1)
                finger = int(m_p.group(2))
                verify = m_p.group(3)
                cond = m_p.group(4)
                part = m_p.group(5)
            elif (m_sp):
                person = m_sp.group(1)
                finger = int(m_sp.group(2))
                verify = m_sp.group(3)
                part = m_sp.group(4)
            elif (m_e):
                person = m_e.group(1)
                finger = int(m_e.group(2))

            # replace
            person = person.replace("\\", "")
            person = person.replace('np', "")
            person = person.replace("ipp", "")
            person = person.replace("org", "")
            person = person.replace("md1", "")
            person = person.replace("md2", "")
            person = person.replace("sdk", "")
            person = person.replace("full", "")
            person = person.replace("normal", "")
            person = person.replace("_", "")
            person = person.replace("pcopy", "")
            person = person.replace("copy", "")
            person = person.replace("md3", "")

            fr = 1
            if info0['match'] == "1":
                fr = 0

            writer.writerow(
                [info0['enroll'], info0['verify'], info0['match'], info0['score'], info0['rotation'], info0['dx'],
                 info0['dx'], enroll_path, verify_path, person, finger, verify, quality, cond, part, fr])

            # # copy image
            # if verify_path.find('_md3') >= 0 and info0['match'] == '1':
            #     img_path = os.verify_path.join(root_folder, verify_path)
            #     out_path = os.verify_path.join(root_folder, "md3", verify_path)
            #     if not os.verify_path.exists(os.verify_path.dirname(out_path)):
            #         os.makedirs(os.verify_path.dirname(out_path))
            #     copyfile(img_path, out_path)
            # elif info0['match'] == '0':
            #     img_path = os.verify_path.join(root_folder, verify_path)
            #     out_path = os.verify_path.join(root_folder, "fail", verify_path)
            #     if not os.verify_path.exists(os.verify_path.dirname(out_path)):
            #         os.makedirs(os.verify_path.dirname(out_path))
            #     copyfile(img_path, out_path)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path0", help="directory to parse")
    parser.add_argument("path1", help="directory to parse")
    parser.add_argument("filepath", help="directory to parse", default="")
    args = parser.parse_args()

    gen_file = args.path0
    index_file = args.path1
    output_file = args.filepath

    gen_data0 = parse_genuines(gen_file)
    index_data0 = parse_index(index_file)

    analysis_info(gen_data0, index_data0, output_file)
    pass
