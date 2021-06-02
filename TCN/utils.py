import torch

class TensorQueue(object):
    def __init__(self, tensor):
        self.tensor = tensor # shape [B, CH, IN]
    
    def __call__(self):
        return self.tensor
    
    def push(self, val):
        # self.tensor = torch.cat((self.tensor[:,:,1:], val[:,:,:]), dim=2)
        self.tensor[:val.size()[0],:,:] = torch.cat((self.tensor[:val.size()[0],:,1:], val[:,:,:]), dim=2)


class CircularQueue(object):
    def __init__(self, tensor):
        self.tensor = tensor # shape [B, CH, IN]
        self.index = torch.arange(self.tensor.size()[2]).to(tensor.device)
        self.roll = 0

    def __call__(self):
        idx = (self.index + self.roll) % self.tensor.size()[2]
        return self.tensor.index_select(2, idx)
    
    def push(self, val):
        self.tensor[:val.size()[0],:,self.roll] = val.squeeze(2)
        self.roll = (self.roll + 1) % self.tensor.size()[2]