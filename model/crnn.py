import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.embedding = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, input):
        output, _ = self.rnn(input)
        seq_len, batch_size, dim = output.size()    # dim == hidden_dim * 2
        output = output.view(seq_len * batch_size, dim)
        output = self.embedding(output)
        output = output.view(seq_len, batch_size, -1)
        return output


class CRNN(nn.Module):
    def __init__(self, hidden_dim, class_count, in_channels=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.class_count = class_count
        self.in_channels = in_channels
        self.cnn = self._init_cnn()
        self.rnn = self._init_rnn()
        self._init_weights()
    
    def _init_cnn(self):
        cnn = nn.Sequential()
        ks = (3, 3, 3, 3, 3, 3, 2)  # kernel size
        ss = (1, 1, 1, 1, 1, 1, 1)  # stride size
        ps = (1, 1, 1, 1, 1, 1, 0)  # padding size
        channels = (64, 128, 256, 256, 512, 512, 512)

        def add_conv(i, batch_norm=False):
            in_channels = self.in_channels if i == 0 else channels[i - 1]
            out_channels = channels[i]
            cnn.add_module(f'conv{i+1}', nn.Conv2d(in_channels, out_channels, ks[i], ss[i], ps[i]))
            if batch_norm:
                cnn.add_module(f'batch_norm{i+1}', nn.BatchNorm2d(out_channels))
            cnn.add_module(f'relu{i+1}', nn.ReLU(True))

        add_conv(0)
        cnn.add_module(f'pooling{1}', nn.MaxPool2d(2, 2))
        add_conv(1)
        cnn.add_module(f'pooling{2}', nn.MaxPool2d(2, 2))
        add_conv(2, True)
        add_conv(3)
        cnn.add_module(f'pooling{3}', nn.MaxPool2d((2, 1), (2, 1), (0, 1)))
        add_conv(4, True)
        add_conv(5)
        cnn.add_module(f'pooling{4}', nn.MaxPool2d((2, 1), (2, 1), (0, 1)))
        add_conv(6, True)

        return cnn

    def _init_rnn(self):
        return nn.Sequential(
            BidirectionalLSTM(512, self.hidden_dim, self.hidden_dim),
            BidirectionalLSTM(self.hidden_dim, self.hidden_dim, self.class_count)
        )
    
    def _init_weights(self):
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight.data, mean=0, std=0.02)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal(m.weight.data, mean=1, std=0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        output = self.cnn(input)    # (N, C, H, W)
        assert output.shape[2] == 1, 'the height of feature map must be 1.'
        output = output.squeeze(2)  # (N, C, W)
        output = output.permute(2, 0, 1)
        output = self.rnn(output)
        return output
