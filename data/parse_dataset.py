import os.path as osp


def parse_mjsynth(dir, file):
    assert isinstance(file, str) or isinstance(file, list)

    image_path_list = list()
    label_list = list()

    def parse_one_file(file):
        with open(osp.join(dir, file), 'r') as f:
            for line in f.readlines():
                line = line.strip()[2:].split(' ')[0] # remove './' and ' xxxxx'

                label = line.split('_')[1]
                label_list.append(label)

                image_path = osp.join(dir, line)
                image_path_list.append(image_path)

    if isinstance(file, str):
        parse_one_file(file)
    elif isinstance(file, list):
        for f in file:
            parse_one_file(f)

    return image_path_list, label_list
