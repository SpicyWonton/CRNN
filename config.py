# dataset
MJSYNTH_DIR = '/mnt/hdd3/std2021/huyi/datasets/MJSynth/mnt/ramdisk/max/90kDICT32px/'
MJSYNTH_TRAIN_ANNOTATION_FILE = ['annotation_train.txt', 'annotation_test.txt']
MJSYNTH_VAL_ANNOTATION_FILE = 'annotation_val.txt'
MJSYNTH_LMDB_DIR = '/mnt/hdd3/std2021/huyi/datasets/lmdb/mjsynth/'

# train
BATCH_SIZE = 64
NUM_WORKERS = 2
IMG_HEIGHT = 32
IMG_WIDTH = 100
ALPHABET = '0123456789abcdefghijklmnopqrstuvwxyz'
HIDDEN_DIM = 256    # size of the lstm hidden state
CHECKPOINT_DIR = '/mnt/hdd3/std2021/huyi/checkpoints/CRNN/'
CHECKPOINT_FILE = ''
START_EPOCH = 0
EPOCH_COUNT = 25
BEST_ACCURACY = 0
SEED = 996
