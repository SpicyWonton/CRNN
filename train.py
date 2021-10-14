import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

import config
import utils
from data import LMDBDataset, AlignCollate
from model import CRNN
from utils import CharDict, save_checkpoint, set_random_seed

def main():
    # set random seed
    set_random_seed()

    cudnn.benchmark = True

    # device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # char_dict
    char_dict = CharDict(config.ALPHABET)

    # dataset and dataloader
    train_dataset = LMDBDataset(config.MJSYNTH_LMDB_DIR, 'train')
    train_dataloader = DataLoader(train_dataset, config.BATCH_SIZE, True, 
        num_workers=config.NUM_WORKERS, 
        collate_fn=AlignCollate(config.IMG_HEIGHT, config.IMG_WIDTH), drop_last=True)
    val_dataset = LMDBDataset(config.MJSYNTH_LMDB_DIR, 'val')
    val_dataloader = DataLoader(val_dataset, config.BATCH_SIZE, True, 
        num_workers=config.NUM_WORKERS,
        collate_fn=AlignCollate(config.IMG_HEIGHT, config.IMG_WIDTH), drop_last=True)

    # model and optimizer
    class_count = len(config.ALPHABET) + 1
    model = CRNN(config.HIDDEN_DIM, class_count).to(device)
    optimizer = optim.Adadelta(model.parameters(), rho=0.9)

    if config.CHECKPOINT_FILE != '':
        checkpoint = torch.load(config.CHECKPOINT_FILE)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        config.START_EPOCH = checkpoint['start_epoch']
        config.BEST_ACCURACY = checkpoint['best_accuracy']
    else:
        config.START_EPOCH = 0
        config.BEST_ACCURACY = 0.0

    # criterion
    criterion = nn.CTCLoss(blank=class_count-1).to(device)

    logger = utils.Logger(config.LOG_DIR, 'mjsynth')

    for epoch in range(config.START_EPOCH, config.EPOCH_COUNT):
        train(model, train_dataloader, criterion, optimizer, char_dict, epoch, device, logger)
        accuracy = validate(model, val_dataloader, criterion, char_dict, epoch, device, logger)

        if accuracy > config.BEST_ACCURACY:
            config.BEST_ACCURACY = accuracy
            save_checkpoint(model, optimizer, accuracy, epoch)


def train(model, loader, criterion, optimizer, char_dict, epoch, device, logger):
    avg_loss = 0.0
    start_time = time.time()
    loader_time = 0
    last_loader_time = time.time()
    model.train()
    for i, (images, labels) in enumerate(loader):
        # images(tensor): (N, 1, 32, 100)
        # labels (tuple): (label1, label2, ...)

        # record the data loading time
        curr_loader_time = time.time()
        loader_time += (curr_loader_time - last_loader_time)
        last_loader_time = curr_loader_time

        batch_size = images.shape[0]
        images = images.to(device)

        preds = model(images) # (T, N, class_count)
        preds = preds.log_softmax(2)    # use log_softmax before ctcloss
        input_lengths = torch.IntTensor([preds.shape[0]] * batch_size).to(device)

        targets, target_lengths = char_dict.encode(labels)
        targets, target_lengths = targets.to(device), target_lengths.to(device)

        loss = criterion(preds, targets, input_lengths, target_lengths)
        avg_loss += loss.item()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if i == 0 or (i + 1) % config.DISPLAY_INTERVAL == 0:
            msg = f'Train: [{epoch+1}/{config.EPOCH_COUNT}][{i+1}/{len(loader)}], train loss: {avg_loss/(i+1):.5}'
            logger.log(msg)
    
    avg_loss /= (i + 1)
    end_time = time.time()
    msg = f'Train: [{epoch+1}/{config.EPOCH_COUNT}], train loss: {avg_loss}, batch time: {end_time-start_time}, ' \
        f'dataloader time: {loader_time}'
    logger.log(msg)


def validate(model, loader, criterion, char_dict, epoch, device, logger):
    avg_loss = 0.0
    start_time = time.time()
    loader_time = 0
    last_loader_time = time.time()
    correct_count = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
             # record the data loading time
            curr_loader_time = time.time()
            loader_time += (curr_loader_time - last_loader_time)
            last_loader_time = curr_loader_time

            batch_size = images.shape[0]
            images = images.to(device)
            
            preds = model(images)   # (T, N, class_count)
            preds = preds.log_softmax(2)    # use log_softmax before ctcloss
            input_lengths = torch.IntTensor([preds.shape[0]] * batch_size).to(device)

            targets, target_lengths = char_dict.encode(labels)
            targets, target_lengths = targets.to(device), target_lengths.to(device)

            loss = criterion(preds, targets, input_lengths, target_lengths)
            avg_loss += loss.item()

            # calculate accuracy
            _, pred_chars = preds.max(2)
            # (T, N) ==> (N, T) ==> (N*T, )
            pred_chars = pred_chars.transpose(1, 0).contiguous().view(-1)
            decoded_words = char_dict.decode(pred_chars, input_lengths) # list of str
            for pred_word, label_word in zip(decoded_words, labels):
                if pred_word == label_word.lower():
                    correct_count += 1
            
            if i == 0 or (i + 1) % config.DISPLAY_INTERVAL == 0:
                msg = f'Validation: [{epoch+1}/{config.EPOCH_COUNT}][{i+1}/{len(loader)}], val loss: {avg_loss/(i+1):.5}'
                logger.log(msg)
                # in order to display
                raw_decoded_words = char_dict.decode(pred_chars, input_lengths, raw=True)[:5] # list of str
                for raw_pred_word, pred_word, label_word in zip(raw_decoded_words, decoded_words, labels):
                    msg = f'Raw pred word: {raw_pred_word} ==> pred word: {pred_word}, label word: {label_word}'
                    logger.log(msg)

        avg_loss /= (i + 1)
        accuracy = correct_count / (config.BATCH_SIZE * (i + 1))
        end_time = time.time()
        msg = f'Validation: [{epoch+1}/{config.EPOCH_COUNT}], accuracy: {accuracy:.5}, val loss: {avg_loss}, ' \
            f'batch_time: {end_time - start_time}, dataloader time: {loader_time}'
        logger.log(msg)
        
        return accuracy


if __name__ == '__main__':
    main()
