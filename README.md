## CRNN

This repo implements [CRNN](https://arxiv.org/abs/1507.05717) in pytorch 1.8, official code can be found at [https://github.com/bgshih/crnn](https://github.com/bgshih/crnn).

### Run demo

Download the pretrained model at [BaiduNetdisk](https://pan.baidu.com/s/1089sSEkctPKtIfBAuyheJg)(fetch code: 1234), put it somewhere in your disk, and change CHECKPOINT_FILE in config.py corespondingly. Then run the demo.py by:

```
$ python demo.py demo/display.png
Raw predicted word: d----ii-s---p---l--a---y--, predicted word: display
```

### Result

I use [MJSynth](https://thor.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz) to train and validate, use [SVT](http://vision.ucsd.edu/~kai/svt/svt.zip) to test. Use the checkpoint at epoch 19,  the accuracy reaches 80.371, compared to 80.8 in original paper.

### Train

Down [MJSynth](https://thor.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz) and [SVT](http://vision.ucsd.edu/~kai/svt/svt.zip), put them somewhere in your disk, and change all the xxx_DIR in config.py. Then train the model by:

```
cd data
python create_lmdb.py # It only needs to be executed once
cd ..
python train.py
```

### Reference

[An old version pytorch implementation](https://github.com/meijieru/crnn.pytorch)





