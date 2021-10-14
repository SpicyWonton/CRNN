import argparse

import cv2
import torch
from torchvision.transforms import transforms

import config
from model import CRNN
from utils import CharDict


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('img_path', type=str, help='the path of the image to recognize')
    args = arg_parser.parse_args()

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    img = cv2.imread(args.img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 32))
    img = transforms.ToTensor()(img).unsqueeze(0).to(device)

    class_count = len(config.ALPHABET) + 1
    model = CRNN(config.HIDDEN_DIM, class_count).to(device)
    if config.CHECKPOINT_FILE != '':
        checkpoint = torch.load(config.CHECKPOINT_FILE)
        model.load_state_dict(checkpoint['model'])
    else:
        raise RuntimeError('Checkpoint must be specified.')

    char_dict = CharDict(config.ALPHABET)

    pred = model(img)   # (T, N, class_count)
    input_lengths = torch.IntTensor([pred.shape[0]] * pred.shape[1]).to(device)
    _, pred_chars = pred.max(2) # (T, N)
    # (T, N) ==> (N, T) ==> (N*T, )
    pred_chars = pred_chars.transpose(1, 0).contiguous().view(-1)
    decoded_word = char_dict.decode(pred_chars, input_lengths) # str
    raw_decoded_word = char_dict.decode(pred_chars, input_lengths, True)

    print(f'Raw predicted word: {raw_decoded_word}, predicted word: {decoded_word}')


if __name__ == '__main__':
    main()
