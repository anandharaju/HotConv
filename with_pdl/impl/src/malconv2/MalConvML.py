from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from malconv2.LowMemConv import LowMemConvBase


def getParams():
    #Format for this is to make it work easily with Optuna in an automated fashion.
    #variable name -> tuple(sampling function, dict(sampling_args) )
    params = {
        'channels'     : ("suggest_int", {'name':'channels', 'low':32, 'high':1024}),
        'log_stride'   : ("suggest_int", {'name':'log2_stride', 'low':2, 'high':9}),
        'window_size'  : ("suggest_int", {'name':'window_size', 'low':32, 'high':256}),
        'layers'       : ("suggest_int", {'name':'layers', 'low':1, 'high':6}),
        'embd_size'    : ("suggest_int", {'name':'embd_size', 'low':4, 'high':64}),
    }
    return OrderedDict(sorted(params.items(), key=lambda t: t[0]))

def initModel(**kwargs):
    new_args = {}
    for x in getParams():
        if x in kwargs:
            new_args[x] = kwargs[x]
            
    return MalConvML(**new_args)


class MalConvML(LowMemConvBase):
    
    def __init__(self, chunk_size, channels, window_size, stride, out_size=2, layers=1, embd_size=8, log_stride=None):
        super(MalConvML, self).__init__(chunk_size=chunk_size, num_filters=channels)

        # ##############################################################
        # To use same embedding weights for all models,
        # use the _weight parameter to supply the custom weights.
        # Shape of weights: (257, 8)
        # ##############################################################
        # CUSTOM_WEIGHT_INIT = torch.nn.init.normal(torch.empty(257, 8)))
        # self.embd = nn.Embedding(257, self.embd_dim, padding_idx=0, _weight=CUSTOM_WEIGHT_INIT)
        self.embd = nn.Embedding(257, 8, padding_idx=0)
        self.embd.weight.requires_grad=False

        if not log_stride is None:
            stride = 2**log_stride
        
        self.convs = nn.ModuleList([nn.Conv1d(embd_size, channels*2, window_size, stride=stride, bias=True)] + [nn.Conv1d(channels, channels*2, window_size, stride=1, bias=True) for i in range(layers-1)])
        self.convs_1 = nn.ModuleList([nn.Conv1d(channels, channels, 1, bias=True) for i in range(layers)])
        self.conv_1 = nn.Conv1d(8, channels, window_size, stride=512, bias=True)
        self.conv_2 = nn.Conv1d(8, channels, window_size, stride=512, bias=True)
        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)

    def processRange(self, x, ghandle, print_gpu_use=False):
        x = self.embd(x)
        # x = torch.transpose(x, -1, -2)
        x = x.permute(0,2,1).contiguous()
        cnn_value = self.conv_1(x)
        cnn_value = torch.relu(cnn_value)
        gating_weight = self.conv_2(x)
        del x
        gating_weight = torch.sigmoid(gating_weight)
        x = cnn_value * gating_weight
        return x
    
    def forward(self, x, ghandle):
        post_conv = x = self.seq2fix(x, ghandle)
        penult = x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x, penult, post_conv
