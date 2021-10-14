# dataset
MJSYNTH_DIR = '/mnt/hdd3/std2021/huyi/datasets/MJSynth/mnt/ramdisk/max/90kDICT32px/'
MJSYNTH_TRAIN_ANNOTATION_FILE = ['annotation_train.txt', 'annotation_test.txt']
MJSYNTH_VAL_ANNOTATION_FILE = 'annotation_val.txt'
MJSYNTH_LMDB_DIR = '/mnt/hdd3/std2021/huyi/datasets/lmdb/mjsynth/'

SVT_DIR = '/mnt/hdd3/std2021/huyi/datasets/svt1/'
SVT_TEST_XML_FILE = 'test.xml'
SVT_LMBD_DIR = '/mnt/hdd3/std2021/huyi/datasets/lmdb/svt/'

# train
BATCH_SIZE = 256
NUM_WORKERS = 2
IMG_HEIGHT = 32
IMG_WIDTH = 100
ALPHABET = '0123456789abcdefghijklmnopqrstuvwxyz'
HIDDEN_DIM = 256    # size of the lstm hidden state
CHECKPOINT_DIR = '/mnt/hdd3/std2021/huyi/checkpoints/CRNN/'
CHECKPOINT_FILE = '/mnt/hdd3/std2021/huyi/checkpoints/CRNN/crnn_19_0.93613.pth'
START_EPOCH = 0
EPOCH_COUNT = 25
DISPLAY_INTERVAL = 500
BEST_ACCURACY = 0
SEED = 996
LOG_DIR = '/home/std2021/huyi/codes/python/MyProjects/CRNN/logs/'
