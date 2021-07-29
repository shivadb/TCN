import torch
import torch.nn as nn
import torch.nn.functional as F
from TCN.tcn import TemporalConvNet
from TCN.tcn_inference import TCNInferenceNet
from TCN.tcn_trt import TCNTRTNet

    
def trt_argmax(x, dim=None, keepdim=False):
    return torch.argmax(x, dim, keepdim).int()

class TCN(nn.Module):
    def __init__(self, input_size=1, output_size=10, num_channels=None, kernel_size=7, dropout=0.05, inference=False, trt=False, apply_max=False):
        super(TCN, self).__init__()
        num_channels = [25] * 8 if num_channels is None else num_channels
        if inference:
            self.tcn = TCNInferenceNet(input_size, num_channels, kernel_size=kernel_size)
        elif trt:
            self.tcn = TCNTRTNet(input_size, num_channels, kernel_size=kernel_size)
        else:
            self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.receptive_field = self.tcn.receptive_field
        self.inference_mode = inference
        self.is_trt = trt
        self.apply_max = apply_max

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        if not self.inference_mode:
            y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
            if self.is_trt:
                # torch2trt converters do not support slicing operations well
                y1 =  torch.narrow(y1, 2, y1.size()[2]-1, 1).squeeze(2)
                o = self.linear(y1)

                # torch2trt has not implemented a converter for F.log_softmax
                logits = F.softmax(o, dim=1)

                if self.apply_max:
                    # torch2trt also does not have an implementation for torch.argmax
                    return trt_argmax(logits, dim=1)
                    # return torch.topk(logits, 1, dim=1)
                    # return torch.max(logits, dim=1, keepdim=True)
                else:
                    return logits
            else: 
                o = self.linear(y1[:, :, -1])
                logits = F.log_softmax(o, dim=1)

                if self.apply_max:
                    return torch.argmax(logits, dim=1)
                else:
                    return logits

        else:
            # inference_mode is obsolete as onnx and tensorrt engines do not support
            # stateful modules
            y1 = self.tcn(inputs).squeeze(dim=2) # (N, C, 1)
            o = self.linear(y1)
            logits = F.log_softmax(o, dim=1)

            if self.apply_max:
                return torch.argmax(logits, dim=1)
            else:
                return logits

    # def inference(self, input):
    #     return F.log_softmax(self.linear(self.tcn.inference(input).squeeze(dim=2)), dim=1)
    
    # def set_fast_inference(self, batch_size):
    #     if not self.inference_mode:
    #         self.tcn.set_fast_inference(batch_size)
    #         self.inference_mode = True

    def compare(self, inputs):
        y1 = self.tcn(inputs)
        y2 = torch.zeros(y1.size()).to(y1.device)
        for i in range(inputs.size()[2]):
            y2[:,:,i] = self.tcn.inference(inputs[:,:,i].view(inputs.size()[0], inputs.size()[1], 1)).squeeze()
        
        return (y1 == y2).all().item()
    
    def reset_cache(self):
        self.tcn.reset_cache()
        
