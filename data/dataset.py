import io
import os.path as osp
import sys
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LMDBDataset(Dataset):
    def __init__(self, lmdb_dir, flag='train', transform=None, target_transform=None):
        super().__init__()
        assert flag == 'train' or flag == 'val', \
            'flag must be "train" or "val"'

        if flag == 'train':
            lmdb_dir = osp.join(lmdb_dir, 'train/')
        elif flag == 'val':
            lmdb_dir = osp.join(lmdb_dir, 'val/')
        
        self.env = lmdb.open(lmdb_dir, readonly=True, readahead=False, max_readers=1, lock=False)
        if not self.env:
            print(f'can not create lmdb from {lmdb_dir}.')
            sys.exit(0)
        
        with self.env.begin(write=False) as txn:
            self.sample_count = int(txn.get('sample_count'.encode()).decode())
        
        self.transform = transform
        self.target_transform = transform
    
    def __len__(self):
        return self.sample_count
    
    def __getitem__(self, index):
        assert index <= len(self), 'index range error.'

        index += 1
        with self.env.begin(write=False) as txn:
            image_key = f'image-{index:09}'
            label_key = f'label-{index:09}'

            image_bytes = txn.get(image_key.encode())
            image_byte_stream = io.BytesIO(image_bytes)
            image = Image.open(image_byte_stream).convert('L')
            label = str(txn.get(label_key.encode()).decode())

            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return image, label


class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.to_tensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class AlignCollate(object):
    def __init__(self, img_height=32, img_width=100, keep_ratio=False, min_ratio=1):
        self.img_height = img_height
        self.img_width = img_width
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        # batch is a list of tuples: [(image1, label1), (image2, label2), ...]
        # images and labels are tuples: (image1, image2, ...), (label1, label2, ...)
        images, labels = zip(*batch)    

        img_height = self.img_height
        img_width = self.img_width
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            img_width = int(np.floor(max_ratio * img_height))
            img_width = max(img_height * self.min_ratio, img_width)  # assure img_height <= img_width

        transform = ResizeNormalize((img_width, img_height))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return (images, labels)
