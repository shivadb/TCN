import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
# from .utils import TensorQueue, CircularQueue

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# class TensorCache(nn.Module):
#     def __init__(self, tensor):
#         super(TensorCache, self).__init__()
#         self.cache = tensor # shape [B, CH, IN]
#         self.index = torch.arange(self.cache.size()[2]).to(tensor.device)
#         self.roll = 0
#         self.input_size = tensor.size()[2]
    
#     def forward(self, x):
#         self.cache[:x.size()[0],:,self.roll] = x.squeeze(2).detach()
#         self.roll = (self.roll + 1) % self.input_size
#         idx = (self.index + self.roll) % self.input_size
#         return self.cache.index_select(2, idx)


class TensorCache(nn.Module):
    def __init__(self, tensor):
        super(TensorCache, self).__init__()
        self.cache = tensor # shape [B, CH, IN]
    
    def forward(self, x):
        cache_update = torch.cat((self.cache[:x.size()[0],:,1:], x[:,:,:].detach()), dim=2)
        self.cache = torch.cat((cache_update, self.cache[x.size()[0]:, :, :]), dim=0)
        # self.cache[:x.size()[0],:,:] = torch.cat((self.cache[:x.size()[0],:,1:], x[:,:,:].detach()), dim=2)
        return self.cache


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

    def set_fast_inference(self, batch_size):
        device = next(self.parameters()).device

        self.fast_conv1 = weight_norm(nn.Conv1d(self.conv1.in_channels, self.conv1.out_channels, 
                                           self.conv1.kernel_size, stride=self.conv1.stride, 
                                           padding=0, dilation=self.conv1.dilation)).to(device)
        self.fast_conv1.load_state_dict(self.conv1.state_dict())

        self.fast_conv2 = weight_norm(nn.Conv1d(self.conv2.in_channels, self.conv2.out_channels, 
                                           self.conv2.kernel_size, stride=self.conv2.stride, 
                                           padding=0, dilation=self.conv2.dilation)).to(device)
        self.fast_conv2.load_state_dict(self.conv2.state_dict())

        self.cache1 = TensorCache(torch.zeros(
            batch_size, 
            self.fast_conv1.in_channels, 
            (self.fast_conv1.kernel_size[0]-1)*self.fast_conv1.dilation[0] + 1
            ).to(device))
        
        self.cache2 = TensorCache(torch.zeros(
            batch_size, 
            self.fast_conv2.in_channels, 
            (self.fast_conv2.kernel_size[0]-1)*self.fast_conv2.dilation[0] + 1
            ).to(device))

        self.stage1 = nn.Sequential(self.fast_conv1, self.relu1)
        self.stage2 = nn.Sequential(self.fast_conv2, self.relu2)

    def reset_cache(self):
        device = next(self.parameters()).device
        self.cache1.cache = torch.zeros(self.cache1.cache.size()).to(device)
        self.cache2.cache = torch.zeros(self.cache2.cache.size()).to(device)


    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    
    def inference(self, x):
        '''
        x is of shape (B, CH, 1)
        '''
        out = self.stage1(self.cache1(x)[:x.size()[0], :, :])
        out = self.stage2(self.cache2(out)[:x.size()[0], :, :])

        res = x if self.downsample is None else self.downsample(x)
        # print(f'\t res shape: {res.size()}')

        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        self.receptive_field = 1
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
            self.receptive_field += 2*(kernel_size-1)*dilation_size
        self.network = nn.Sequential(*layers)

    def set_fast_inference(self, batch_size):
        for block in self.network:
            block.set_fast_inference(batch_size)

    def forward(self, x):
        return self.network(x)

    def inference(self, x):
        out = x
        for i, block in enumerate(self.network):
            out = block.inference(out)
            # print(f'out {i} shape: {out.size()}')
            # print()

        return out

    def reset_cache(self):
        for block in self.network:
            block.reset_cache()
