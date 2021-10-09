import os
import os.path as osp

MJSYNTH_DIR = '/mnt/hdd3/std2021/huyi/datasets/MJSynth/mnt/ramdisk/max/90kDICT32px/'
MJSYNTH_ANNOTATION_FILE = 'imlist.txt'


def parse_mjsynth(dir=MJSYNTH_DIR, file=MJSYNTH_ANNOTATION_FILE):
    image_path_list = list()
    label_list = list()

    with open(osp.join(dir, file), 'r') as f:
        for line in f.readlines():
            line = line.strip()[2:] # remove './'

            label = line.split('_')[1]
            label_list.append(label)

            image_path = osp.join(dir, line)
            image_path_list.append(image_path)
    
    return image_path_list, label_list
