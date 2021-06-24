import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
# from .utils import TensorQueue, CircularQueue

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


# class TensorCache(nn.Module):
#     def __init__(self, tensor, name):
#         super(TensorCache, self).__init__()
#         self.cache = tensor # shape [B, CH, IN]
#         self.register_buffer(name, self.cache, persistent=False)
    
#     def forward(self, x):
#         cache_update = torch.cat((self.cache[:,:,1:], x[:,:,:].detach()), dim=2)
#         self.cache = cache_update
#         # self.cache[:x.size()[0],:,:] = torch.cat((self.cache[:x.size()[0],:,1:], x[:,:,:].detach()), dim=2)
#         return self.cache

# class TensorCache(nn.Module):
#     def __init__(self, tensor, name):
#         super(TensorCache, self).__init__()
#         self.cache_name = name # shape [B, CH, IN]
#         self.register_buffer(name, tensor, persistent=False)
    
#     def forward(self, x):
#         cache_update = torch.cat((getattr(self, self.cache_name)[:,:,1:], x[:,:,:].detach()), dim=2)
#         setattr(self, self.cache_name, cache_update)
#         return getattr(self, self.cache_name)

# class TensorCache(torch.jit.ScriptModule):
#     def __init__(self, tensor, name):
#         super(TensorCache, self).__init__()
#         self.register_buffer('cache', tensor)
    
#     @torch.jit.script_method
#     def forward(self, x):
#         cache_update = torch.cat((self.cache[:,:,1:], x[:,:,:].detach()), dim=2)
#         self.cache = cache_update
#         return self.cache

class TensorCache(nn.Module):
    def __init__(self, tensor):
        super(TensorCache, self).__init__()
        self.register_buffer('cache', tensor)
    
    def forward(self, x):
        cache_update = torch.cat((self.cache[:,:,1:], x[:,:,:].detach()), dim=2)
        self.cache = cache_update
        return self.cache


class TemporalInferenceBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, block_num, batch_size=1):
        super(TemporalInferenceBlock, self).__init__()
        self.in_ch, self.k, self.d = n_inputs, kernel_size, dilation

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=0, dilation=dilation))
        self.relu1 = nn.ReLU()

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=0, dilation=dilation))
        self.relu2 = nn.ReLU()

        self.batch_size = batch_size

        self.cache1 = TensorCache(torch.zeros(
            batch_size, 
            self.conv1.in_channels, 
            (self.conv1.kernel_size[0]-1)*self.conv1.dilation[0] + 1
            ))
        
        self.cache2 = TensorCache(torch.zeros(
            batch_size, 
            self.conv2.in_channels, 
            (self.conv2.kernel_size[0]-1)*self.conv2.dilation[0] + 1
            ))
        
        self.stage1 = nn.Sequential(self.conv1, self.relu1)
        self.stage2 = nn.Sequential(self.conv2, self.relu2)


        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()
        self.relu = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if isinstance(self.downsample, nn.modules.conv.Conv1d):
            self.downsample.weight.data.normal_(0, 0.01)

    def reset_cache(self):
        device = next(self.parameters()).device
        self.cache1.cache = torch.zeros(self.cache1.cache.size()).to(device)
        self.cache2.cache = torch.zeros(self.cache2.cache.size()).to(device)


    def forward(self, x):
        '''
        x is of shape (B, CH, 1)
        '''
        # out = self.stage1(self.cache1(x)[:x.size()[0], :, :])
        # out = self.stage2(self.cache2(out)[:x.size()[0], :, :])
        out = self.stage1(self.cache1(x))
        out = self.stage2(self.cache2(out))

        res = self.downsample(x)
        # print(f'\t res shape: {res.size()}')

        return self.relu(out + res)
    


class TCNInferenceNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2):
        super(TCNInferenceNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        self.receptive_field = 1
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalInferenceBlock(in_channels, out_channels, kernel_size, stride=1, 
                                                dilation=dilation_size, batch_size=1, block_num=i)]
            self.receptive_field += 2*(kernel_size-1)*dilation_size
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = x
        for i, block in enumerate(self.network):
            out = block(out)
            # print(f'out {i} shape: {out.size()}')
            # print()

        return out

    def reset_cache(self):
        for block in self.network:
            block.reset_cache()
