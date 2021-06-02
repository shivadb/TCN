import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from .utils import TensorQueue

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.in_ch, self.k, self.d = n_inputs, kernel_size, dilation

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def fast_inference(self):
        self.fast_conv1 = weight_norm(nn.Conv1d(self.conv1.in_channels, self.conv1.out_channels, 
                                           self.conv1.kernel_size, stride=self.conv1.stride, 
                                           padding=0, dilation=self.conv1.dilation))
        self.fast_conv1.load_state_dict(self.conv1.state_dict())

        self.fast_conv2 = weight_norm(nn.Conv1d(self.conv2.in_channels, self.conv2.out_channels, 
                                           self.conv2.kernel_size, stride=self.conv2.stride, 
                                           padding=0, dilation=self.conv2.dilation))
        self.fast_conv2.load_state_dict(self.conv2.state_dict())

        self.fast_net = nn.Sequential(self.fast_conv1, self.relu1, self.fast_conv2, self.relu2)


    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    
    def single_forward(self, x):
        print(f'x shape: {x.size()}')
        out = self.fast_net(x)
        print(f'out shape: {out.size()}')
        res = x if self.downsample is None else self.downsample(x)
        print(f'res shape: {res.size()}')

        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.cache = None
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def fast_inference(self):
        for block in self.network:
            block.fast_inference()

    def forward(self, x):
        return self.network(x)

    def single_forward(self, x):
        device = next(self.parameters()).device
        # print(f'x shape: {x.size()}')
        # print(f'x: {x}')

        if not self.cache:
            self.cache = [TensorQueue(torch.zeros(x.size()[0], l.in_ch, (l.k-1)*l.d + 1).to(device)) for l in self.network]
        
        # print(f'Initial Cache 0: {self.cache[0]()}')
        # self.cache[0].push(x)
        # print(f'Cache 0 after push: {self.cache[0]()}')

        out = x
        for i, block in enumerate(self.network):
            self.cache[i].push(out)
            out = block.single_forward(self.cache[i]())
            print(f'out {i} shape: {out.size()}')
            break

