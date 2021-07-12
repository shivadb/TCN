import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class TrimTensor(nn.Module):
    def __init__(self, trim_dim, trim_size):
        super(TrimTensor, self).__init__()
        self.trim_dim = trim_dim
        self.trim_size = trim_size
    
    def forward(self, x):
        l = x.size()[self.trim_dim]
        return torch.narrow(x, self.trim_dim, 0, l-self.trim_size)

class TemporalTRTBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalTRTBlock, self).__init__()
        self.in_ch, self.k, self.d = n_inputs, kernel_size, dilation

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        self.padding = padding
        self.trim = TrimTensor(2, padding)
        self.relu1 = nn.ReLU()

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        
        self.relu2 = nn.ReLU()

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)


    def forward(self, x):
        # s1 = self.relu1(self.conv1(x)[:,:,:-self.padding])
        # out = self.relu2(self.conv2(s1)[:,:,:-self.padding])
        s1 = self.relu1(self.trim(self.conv1(x)))
        out = self.relu2(self.trim(self.conv2(s1)))
        # out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNTRTNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2):
        super(TCNTRTNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        self.receptive_field = 1
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalTRTBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]
            self.receptive_field += 2*(kernel_size-1)*dilation_size
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
