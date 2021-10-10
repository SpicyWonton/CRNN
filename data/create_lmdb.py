import os
import os.path as osp
import sys

import cv2
import lmdb
import numpy as np

sys.path.append('..')
from config import MJSYNTH_DIR, MJSYNTH_TRAIN_ANNOTATION_FILE, \
    MJSYNTH_VAL_ANNOTATION_FILE, MJSYNTH_LMDB_DIR


def _check_image_is_valid(image_bytes):
    if image_bytes is None:
        return False
    
    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    if image_buffer.shape[0] == 0:
        return False

    image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if image is None:
        return False
    
    image_height, image_width = image.shape[0], image.shape[1]
    if image_height * image_width == 0:
        return False
    
    return True


def _write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if isinstance(v, bytes):
                txn.put(k.encode(), v)
            else:
                txn.put(k.encode(), v.encode())


def create_lmdb(lmdb_dir, image_path_list, label_list, flag='train', lexicon_list=None):
    assert len(image_path_list) == len(label_list)
    assert flag == 'train' or flag == 'val', \
        'flag must be "train" or "val"'
    
    if flag == 'train':
        lmdb_dir = osp.join(lmdb_dir, 'train/')
    elif flag == 'val':
        lmdb_dir = osp.join(lmdb_dir, 'val/')
    if not osp.exists(lmdb_dir):
        os.makedirs(lmdb_dir)
    env = lmdb.open(lmdb_dir, map_size=1099511627776)

    cache = dict()
    count = 1
    sample_count = len(image_path_list)
    invalid_count = 0

    for i in range(sample_count):
        image_path = image_path_list[i]
        label = label_list[i]
        
        if not os.path.exists(image_path):
            print(f'{image_path} does not exist.')
            continue
        
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            if not _check_image_is_valid(image_bytes):
                invalid_count += 1
                print(f"{image_path} is invalid.")
                continue
        
        image_key = f'image-{count:09}'
        label_key = f'label-{count:09}'
        cache[image_key] = image_bytes
        cache[label_key] = label
        if lexicon_list:
            lexicon_key = f'lexicon-{count:09}'
            cache[lexicon_key] = ' '.join(lexicon_list[i])
        
        if count % 10000 == 0:
            _write_cache(env, cache)
            cache = dict()
            print(f'({count}/{sample_count}) caches have been written.')
        count += 1
    
    sample_count = count - 1
    cache['sample_count'] = str(sample_count)
    _write_cache(env, cache)
    env.close()
    print(f'The lmdb created has {sample_count} samples.')
    print(f'Removed {invalid_count} invalid image(s).')


if __name__ == '__main__':
    from parse_dataset import parse_mjsynth

    train_image_path_list, train_label_list = parse_mjsynth(MJSYNTH_DIR, 
        MJSYNTH_TRAIN_ANNOTATION_FILE)
    create_lmdb(MJSYNTH_LMDB_DIR, train_image_path_list, train_label_list, 'train')

    val_image_path_list, val_label_list = parse_mjsynth(MJSYNTH_DIR,
        MJSYNTH_VAL_ANNOTATION_FILE)
    create_lmdb(MJSYNTH_LMDB_DIR, val_image_path_list, val_label_list, 'val')
