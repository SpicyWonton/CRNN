import io
import sys
from PIL import Image

import lmdb
from torch.utils.data import Dataset, dataset


class LMDBDataset(Dataset):
    def __init__(self, lmdb_dir, transform=None, target_transform=None):
        super().__init__()
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
