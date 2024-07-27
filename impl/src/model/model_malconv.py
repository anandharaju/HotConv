import torch
import torch.nn as nn


class Multiply(nn.Module):
  def __init__(self):
    super(Multiply, self).__init__()

  def forward(self, x, y):
    t = x * y
    return t


class MalConv(nn.Module):

    def __init__(self, args):
        super(MalConv, self).__init__()
        self.k = 1
        self.device = args.device
        self.max_len = args.max_len
        self.window_size = args.window_size
        self.batch_size = args.batch_size
        self.num_filters = args.num_filters
        self.embd_dim = 8

        # ##############################################################
        # To use same embedding weights for all models,
        # use the _weight parameter to supply the custom weights.
        # Shape of weights: (257, 8)
        # ##############################################################
        # CUSTOM_WEIGHT_INIT = torch.nn.init.normal(torch.empty(257, 8)))
        # self.embd = nn.Embedding(257, self.embd_dim, padding_idx=0, _weight=CUSTOM_WEIGHT_INIT)
        self.embd = nn.Embedding(257, self.embd_dim, padding_idx=0)
        self.stride = args.stride
        self.conv_1 = nn.Conv1d(self.embd_dim, args.num_filters, args.window_size, stride=args.stride, bias=True)
        self.conv_2 = nn.Conv1d(self.embd_dim, args.num_filters, args.window_size, stride=args.stride, bias=True)
        self.pooling = nn.AdaptiveMaxPool1d(self.k, return_indices=True)
        
        self.fc_1 = nn.Linear(self.num_filters * self.k, self.num_filters * self.k)
        self.fc_2 = nn.Linear(self.num_filters * self.k, args.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.glu = nn.GLU(dim=1)
        # Using a multiplication operator would not show the actual memory consumption in torchsummary
        # Hence defining it as a layer
        self.mul = Multiply()
        
        
        self.transformer = nn.Transformer(d_model=self.num_filters, nhead=8, num_encoder_layers=64, num_decoder_layers=0, batch_first=True)
        

    def forward(self, x):
        x = self.embd(x.long())
        # x = torch.transpose(x, -1, -2)
        x = x.permute(0,2,1).contiguous()
        cnn_value = self.conv_1(x)
        gating_weight = self.conv_2(x)
        cnn_value = self.relu(cnn_value)
        gating_weight = self.sigmoid(gating_weight)
        # Implementing multiply using basic operator still consumes memory
        # but does not show up in torchsummary
        # hence, defining it as a layer.
        # x = cnn_value * gating_weight
        x = self.mul(cnn_value, gating_weight)
        x, indices = self.pooling(x)
        x = x.view(-1, self.num_filters * self.k)
        # x = self.transformer(x,x)
        
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        return x
