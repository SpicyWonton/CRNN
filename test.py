import torch
from torch.utils.data import DataLoader

import config
import utils
from data import LMDBDataset, AlignCollate
from model import CRNN


def main():
    # device
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # char_dict
    char_dict = utils.CharDict(config.ALPHABET)
    
    # dataset and dataloader
    test_dataset = LMDBDataset(config.SVT_LMBD_DIR, 'test')
    test_dataloader = DataLoader(test_dataset, 1, True, 
        num_workers=config.NUM_WORKERS, 
        collate_fn=AlignCollate(config.IMG_HEIGHT, config.IMG_WIDTH, True))

    # model
    class_count = len(config.ALPHABET) + 1
    model = CRNN(config.HIDDEN_DIM, class_count).to(device)

    if config.CHECKPOINT_FILE != '':
        checkpoint = torch.load(config.CHECKPOINT_FILE)
        model.load_state_dict(checkpoint['model'])
    else:
        raise RuntimeError('Checkpoint must be specified during the test phase.')
    
    test(model, test_dataloader, char_dict, device)


def test(model, loader, char_dict, device):
    model.eval()
    correct_count = 0
    logger = utils.Logger(config.LOG_DIR, 'svt')
    for i, (image, label) in enumerate(loader):
        image = image.to(device)
        batch_size = image.shape[0]
        preds = model(image)    # (T, N, class_count)
        input_lengths = torch.IntTensor([preds.shape[0]] * batch_size).to(device)

        # calculate accuracy
        _, pred_chars = preds.max(2)    # (T, N)
        # (T, N) ==> (N, T) ==> (N*T, )
        pred_chars = pred_chars.transpose(1, 0).contiguous().view(-1)
        decoded_words = char_dict.decode(pred_chars, input_lengths) # str

        if isinstance(decoded_words, list):
            for pred_word, label_word in zip(decoded_words, label):
                logger.log(f'pred word: {pred_word}, label word: {label_word.lower()}')
                if pred_word == label_word.lower():
                    correct_count += 1
        elif isinstance(decoded_words, str):
            logger.log(f'pred word: {decoded_words}, label word: {label[0].lower()}')
            if decoded_words == label[0].lower():
                correct_count += 1
        else:
            raise RuntimeError('decoded_words must be list or str.')

    accuracy = correct_count / (i + 1)
    logger.log(f'Test accuracy: {accuracy:.5}')


if __name__ == '__main__':
    main()
