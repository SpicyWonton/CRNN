import os
import os.path as osp
import xml.etree.ElementTree as ET

import cv2


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


def parse_svt(dir, xml_file):
    image_path_list = list()
    label_list = list()

    cropped_img_dir = osp.join(dir, 'cropped_img/')
    if not osp.exists(cropped_img_dir):
        os.makedirs(cropped_img_dir)
    
    tree = ET.parse(osp.join(dir, xml_file))
    root = tree.getroot()
    for image in root:
        img_name = image[0].text
        uncropped_img = cv2.imread(osp.join(dir, img_name))
        for i, bbox in enumerate(image.find('taggedRectangles')):
            label = bbox[0].text    # the cropped image's content

            h, w, x1, y1 = int(bbox.attrib['height']), int(bbox.attrib['width']), \
                int(bbox.attrib['x']), int(bbox.attrib['y'])
            h, w, x1, y1 = max(h, 0), max(w, 0), max(x1, 0), max(y1, 0) # in one bbox, y=-2
            x2, y2 = x1 + w, y1 + h
            cropped_img = uncropped_img[y1:y2, x1:x2]

            cropped_img_name = img_name.split('/')[1].split('.')[0] + f'_{i}_{label}.jpg'
            cropped_img_path = osp.join(cropped_img_dir, cropped_img_name)
            cv2.imwrite(cropped_img_path, cropped_img)

            image_path_list.append(cropped_img_path)
            label_list.append(label)
    
    return image_path_list, label_list
