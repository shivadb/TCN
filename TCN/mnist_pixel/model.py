import torch
import torch.nn as nn
import torch.nn.functional as F
from TCN.tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size=1, output_size=10, num_channels=None, kernel_size=7, dropout=0.05):
        super(TCN, self).__init__()
        num_channels = [25] * 8 if num_channels is None else num_channels
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)

    def single_forward(self, input):
        return F.log_softmax(self.linear(self.tcn.single_forward(input).squeeze(dim=2)), dim=1)
    
    def fast_inference(self, batch_size):
        self.tcn.fast_inference(batch_size)

    def compare(self, inputs):
        y1 = self.tcn(inputs)
        y2 = torch.zeros(y1.size()).to(y1.device)
        for i in range(inputs.size()[2]):
            y2[:,:,i] = self.tcn.single_forward(inputs[:,:,i].view(inputs.size()[0], inputs.size()[1], 1)).squeeze()
        
        return (y1 == y2).all().item()
    
    def reset_cache(self):
        self.tcn.reset_cache()
        
