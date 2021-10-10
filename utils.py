import collections.abc
import os
import os.path as osp
import random

import numpy as np
import torch

import config


class CharDict(object):
    def __init__(self, alphabet, ignore_case=True):
        super().__init__()
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
            alphabet = list(set(alphabet))  # remove duplicate elements
            alphabet.sort(key=lambda k: k)
            alphabet = ''.join(alphabet)
        alphabet += '-' # add blank
        self.alphabet = alphabet
        self._char_to_digit = dict()
        for i, char in enumerate(alphabet):
            self._char_to_digit[char] = i
    
    def encode(self, text):
        if isinstance(text, str):
            if self._ignore_case:
                text = [self._char_to_digit[char.lower()] for char in text]
            else:
                text = [self._char_to_digit[char] for char in text]
            length = [len(text)]
        elif isinstance(text, collections.abc.Iterable):
            length = [len(t) for t in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        
        return torch.IntTensor(text), torch.IntTensor(length)
    
    def decode(self, text, length, raw=False):
        if len(length) == 1:
            length = length[0]
            assert len(text) == length, \
                f'text length: {len(text)} does not match declared length: {length}'
            if raw:
                tmp = [self.alphabet[i] for i in text]
            else:
                tmp = list()
                last_char = '-'
                for i in text:
                    curr_char = self.alphabet[i]
                    # remove '-' and duplicate chars
                    if curr_char != '-' and last_char != curr_char:
                        tmp.append(curr_char)
                    last_char = curr_char
            
            return ''.join(tmp)
        else:
            assert len(text) == length.sum().item(), \
                f'text length: {len(text)} does not match declared length: {length.sum().item()}'
            decoded_texts = []
            index = 0
            for l in length:
                decoded_texts.append(
                    self.decode(text[index:index+l], torch.IntTensor([l]), raw)
                )
                index += l
        
        return decoded_texts


def set_random_seed():
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)


def save_checkpoint(model, optimizer, best_accuracy, epoch):
    print('Saving checkpoint...')
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_accuracy': best_accuracy,
        'epoch': epoch
    }
    file_name = f'crnn_{epoch}_{best_accuracy:.5}'
    if not osp.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
    torch.save(state, osp.join(config.CHECKPOINT_DIR, file_name))
    print(f'Checkpoint {file_name} was saved successfully.')


def delete_log_file(file):
    if osp.exists(file):
        os.remove(file)


def save_log_file(file, log):
    with open(file, 'a') as f:
        f.write(log + '\n')
        f.close()

if __name__ == '__main__':
    char_dict = CharDict(config.ALPHABET)
    # text, length = char_dict.encode(('abcd', '4321'))
    # print(text)
    print(char_dict.alphabet)
    encoded_text = torch.IntTensor([36, 10, 10, 36, 11, 36, 11, 12, 12, 36, 13, 36,
                                    36, 4, 3, 36, 36, 36, 2, 2, 36, 1, 36, 1, 36])
    length = torch.IntTensor([12, 13])
    raw_decoded_texts = char_dict.decode(encoded_text, length, True)
    for text in raw_decoded_texts:
        print(text)
    decoded_texts = char_dict.decode(encoded_text, length)
    for text in decoded_texts:
        print(text)

