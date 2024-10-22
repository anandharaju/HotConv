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
        
        # RNN
        bidirect = True
        hidden = 1
        self.rnn = nn.RNN(input_size=self.embd_dim, hidden_size=hidden, num_layers=1, nonlinearity='tanh', bias=True, batch_first=True, dropout=0.0, bidirectional=bidirect)

        # LSTM
        self.lstm = nn.LSTM(input_size=self.embd_dim, hidden_size=hidden, num_layers=1, bias=True, batch_first=True, dropout=0.0, bidirectional=bidirect, proj_size=0) #, device=None, dtype=None)
        self.tanh = nn.Tanh()
        self.fc_lstm = nn.Linear(hidden * (1 + int(bidirect)), hidden * (1 + int(bidirect)))
        self.fc2_lstm = nn.Linear(hidden * (1 + int(bidirect)), args.num_classes)

        # GRU
        self.gru = nn.GRU(input_size=self.embd_dim, hidden_size=hidden, num_layers=1, bias=True, batch_first=True, dropout=0.0, bidirectional=bidirect)

        # Transformer
        #self.transformer = nn.Transformer(d_model=self.num_filters, nhead=8, num_encoder_layers=64, num_decoder_layers=0, batch_first=True)

        # Encoder-Decoder
        #self.enc = nn.TransformerEncoder()


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
        
    # RNN
    def forward0(self, x):
        x = self.embd(x.long())
        # x = x.permute(0,2,1).contiguous()
        #x = self.lstm(x)
        # out, (ht, ct) = self.rnn(x)
        out, _ = self.rnn(x)
        out = out[: ,-1]
        out = self.tanh(out)
        
        x = self.fc_lstm(out)
        x = self.relu(x)
        x = self.fc2_lstm(x)
        return x
    
    # LSTM
    def forward1(self, x):
        x = self.embd(x.long())
        # x = x.permute(0,2,1).contiguous()
        #x = self.lstm(x)
        out, _ = self.lstm(x)
        out = out[: ,-1]
        out = self.tanh(out)
        
        x = self.fc_lstm(out)
        x = self.relu(x)
        x = self.fc2_lstm(x)
        return x
    
    # GRU
    def forward2(self, x):
        x = self.embd(x.long())
        # x = x.permute(0,2,1).contiguous()
        #x = self.lstm(x)
        out, _ = self.gru(x)
        out = out[: ,-1]
        out = self.tanh(out)
        
        x = self.fc_lstm(out)
        x = self.relu(x)
        x = self.fc2_lstm(x)
        return x
