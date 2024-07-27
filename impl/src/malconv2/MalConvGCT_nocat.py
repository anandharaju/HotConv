from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from malconv2.LowMemConv import LowMemConvBase
from malconv2.MalConvML import MalConvML

def getParams():
    #Format for this is to make it work easily with Optuna in an automated fashion.
    #variable name -> tuple(sampling function, dict(sampling_args) )
    params = {
        'channels'     : ("suggest_int", {'name':'channels', 'low':32, 'high':1024}),
        'log_stride'   : ("suggest_int", {'name':'log2_stride', 'low':2, 'high':9}),
        'window_size'  : ("suggest_int", {'name':'window_size', 'low':32, 'high':256}),
        'layers'       : ("suggest_int", {'name':'layers', 'low':1, 'high':3}),
        'embd_size'    : ("suggest_int", {'name':'embd_size', 'low':4, 'high':16}),
    }
    return OrderedDict(sorted(params.items(), key=lambda t: t[0]))

def initModel(**kwargs):
    new_args = {}
    for x in getParams():
        if x in kwargs:
            new_args[x] = kwargs[x]
            
    return MalConvGCT(**new_args)


class MalConvGCT(LowMemConvBase):
    
    def __init__(self, chunk_size, channels, window_size, stride, out_size=2, layers=1, embd_size=8, log_stride=None, low_mem=False):
        super(MalConvGCT, self).__init__(chunk_size=chunk_size, num_filters=channels)
        self.low_mem = low_mem

        # ##############################################################
        # To use same embedding weights for all models,
        # use the _weight parameter to supply the custom weights.
        # Shape of weights: (257, 8)
        # ##############################################################
        # CUSTOM_WEIGHT_INIT = torch.nn.init.normal(torch.empty(257, 8)))
        # self.embd = nn.Embedding(257, self.embd_dim, padding_idx=0, _weight=CUSTOM_WEIGHT_INIT)
        self.embd = nn.Embedding(257, embd_size, padding_idx=0)
        self.embd.weight.requires_grad=False

        if not log_stride is None:
            stride = 2**log_stride
        
        self.context_net = MalConvML(chunk_size=chunk_size, channels=channels, out_size=channels, window_size=window_size, stride=stride, layers=layers, embd_size=embd_size)
        self.convs = nn.ModuleList([nn.Conv1d(embd_size, channels*2, window_size, stride=stride, bias=True)] + [nn.Conv1d(channels, channels*2, window_size, stride=1, bias=True) for i in range(layers-1)])
        self.linear_atn = nn.ModuleList([nn.Linear(channels, channels) for i in range(layers)])
        self.convs_share = nn.ModuleList([nn.Conv1d(channels, channels, 1, bias=True) for i in range(layers)])
        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)
        self.sigmoid = nn.Sigmoid()
        
    
    #Over-write the determinRF call to use the base context_net to detemrin RF. We should have the same totla RF, and this will simplify logic significantly. 
    def determinRF(self):
        return self.context_net.determinRF()
    
    def processRange(self, x, gct=None):
        if gct is None:
            raise Exception("No Global Context Given")
        
        x = self.embd(x)
        # x = torch.transpose(x, -1, -2) #
        x = x.permute(0,2,1)
        
        for conv_glu, linear_cntx, conv_share in zip(self.convs, self.linear_atn, self.convs_share):
            x = F.glu(conv_glu(x), dim=1)
            x = F.leaky_relu(conv_share(x))
            B = x.shape[0]
            C = x.shape[1]
            ctnx = torch.tanh(linear_cntx(gct))
            ctnx = torch.unsqueeze(ctnx, dim=2)
            x_tmp = x.view(1,B*C,-1)
            x_tmp = F.conv1d(x_tmp, ctnx, groups=B)
            x_gates = x_tmp.view(B, 1, -1)
            gates = torch.sigmoid( x_gates )
            x = x * gates
        return x
    
    def forward(self, x, ghandle, args):
        if self.low_mem:
            global_context = checkpoint.CheckpointFunction.apply(self.context_net.seq2fix,1, x)
        else:
            global_context, args = self.context_net.seq2fix(x, ghandle, args)
        xt = time.time()
        x = global_context
        x = F.leaky_relu(self.fc_1( x ))
        x = self.fc_2(x)
        # print("FP2 [2] time:", time.time()-xt)
        args.time_fp2 += time.time() - xt
        return x, args
