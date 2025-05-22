import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# TODO: consider weight initialization

class conv_block(nn.Module):
    def __init__(self, in_chan, out_chan, ker_size, stride, pool_size):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, ker_size, stride, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.mpool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        out = self.mpool(x)
        return out

class CRNN(nn.Module):
    """
    input: [Batch, Channels, Frequency, Time]
    output: [Batch, 1]
    """
    def __init__(self, in_chan=1):
        super(CRNN, self).__init__()
        self.conv1 = conv_block(in_chan, 96, ker_size=(5, 5), stride=1, pool_size=(4, 2))
        self.conv2 = conv_block(96, 128, ker_size=(5, 5), stride=1, pool_size=(2, 2))
        self.conv3 = conv_block(128, 128, ker_size=(5, 5), stride=1, pool_size=(5, 2))
        self.rnn1 = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        print(f'1 - {x.shape}')
        x = self.conv2(x)
        print(f'2 - {x.shape}')
        x = self.conv3(x)        # [B,C,H,W] with h=1
        print(f'3 - {x.shape}')
        x = x.squeeze(2)         # [B,C,W]
        print(f'4 - {x.shape}')
        x = x.permute(0, 2, 1)   # [B,W,C]  # batch first = True
        print(f'5 - {x.shape}')
        # RNN
        x, _ = self.rnn1(x)
        print(f'6 - {x.shape}')
        # dense readout
        x = self.dense(x[:, -1, :])
        print(f'7 - {x.shape}')
        x = self.sigmoid(x)
        return x


class CRNN_2(nn.Module):
    """
    input: [Batch, Channels, Frequency, Time]
    output: [Batch, 1]
    """
    def __init__(self, in_chan=1,kernel_size=(5.5), pool_size1=(4,2), pool_size2=(2,2), pool_size3=(2,2), gru=64):
        super(CRNN_2, self).__init__()
        self.conv1 = conv_block(in_chan, 96, ker_size=kernel_size, stride=1, pool_size=pool_size1)
        self.conv2 = conv_block(96, 128, ker_size=kernel_size, stride=1, pool_size=pool_size2)
        self.conv3 = conv_block(128, 128, ker_size=kernel_size, stride=1, pool_size=pool_size3)
        self.rnn1 = nn.GRU(input_size=128, hidden_size=gru, num_layers=2, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)        # [B,C,H,W] with h=1
        x = x.squeeze(2)         # [B,C,W]
        x = x.permute(0, 2, 1)   # [B,W,C]  # batch first = True
        # RNN
        x, _ = self.rnn1(x)
        return x[:, -1, :]
    
class CRNN_2_4(nn.Module):
    """
    input: [Batch, Channels, Frequency, Time]
    output: [Batch, 1]
    """
    def __init__(self, in_chan=1,kernel_size=(5.5), pool_size1=(4,2), pool_size2=(2,2), pool_size3=(2,2), pool_size4=(2,2), gru=64):
        super(CRNN_2_4, self).__init__()
        self.conv1 = conv_block(in_chan, 96, ker_size=kernel_size, stride=1, pool_size=pool_size1)
        self.conv2 = conv_block(96, 128, ker_size=kernel_size, stride=1, pool_size=pool_size2)
        self.conv3 = conv_block(128, 128, ker_size=kernel_size, stride=1, pool_size=pool_size3)
        self.conv4 = conv_block(128, 128, ker_size=kernel_size, stride=1, pool_size=pool_size4)
        self.rnn1 = nn.GRU(input_size=128, hidden_size=gru, num_layers=2, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)        # [B,C,H,W] with h=1
        x = x.squeeze(2)         # [B,C,W]
        x = x.permute(0, 2, 1)   # [B,W,C]  # batch first = True
        # RNN
        x, _ = self.rnn1(x)
        return x[:, -1, :]
    
class CRNN_2_2(nn.Module):
    """
    input: [Batch, Channels, Frequency, Time]
    output: [Batch, 1]
    """
    def __init__(self, in_chan=1,kernel_size=(5.5), pool_size1=(4,2), pool_size2=(2,2), gru=64):
        super(CRNN_2_2, self).__init__()
        self.conv1 = conv_block(in_chan, 96, ker_size=kernel_size, stride=1, pool_size=pool_size1)
        self.conv2 = conv_block(96, 128, ker_size=kernel_size, stride=1, pool_size=pool_size2)
        self.rnn1 = nn.GRU(input_size=128, hidden_size=gru, num_layers=2, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)      # [B,C,H,W] with h=1
        x = x.squeeze(2)         # [B,C,W]
        x = x.permute(0, 2, 1)   # [B,W,C]  # batch first = True
        # RNN
        x, _ = self.rnn1(x)
        return x[:, -1, :]



class Siamese_CRNN(nn.Module):
    """
    input: ([Batch, Channels, Frequency, Time]) x2
    output: ([Batch, 1,128]) x2
    """
    def __init__(self, in_chan=1, kernel_size=(5.5), pool_size1=(4,2), pool_size2=(2,2), pool_size3=(2,2), gru=64):
        super(Siamese_CRNN, self).__init__()
        self.conv1 = conv_block(in_chan, 96, ker_size = kernel_size, stride=1, pool_size=pool_size1)   
        self.conv2 = conv_block(96, 128, ker_size = kernel_size, stride=1, pool_size=pool_size2)
        self.conv3 = conv_block(128, 128, ker_size = kernel_size, stride=1, pool_size=pool_size3)
        self.rnn1 = nn.GRU(input_size=128, hidden_size=gru, num_layers=2, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        # CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)        # [B,C,H,W] with h=1
        x = x.squeeze(2)         # [B,C,W]
        x = x.permute(0, 2, 1)   # [B,W,C]  # batch first = True
        # RNN
        x, _ = self.rnn1(x)
        return x[:, -1, :]

    def forward(self, in_1, in_2):
        out_1 = self.forward_once(in_1)
        out_2 = self.forward_once(in_2)

        return out_1, out_2
    
class Siamese_CRNN_4(nn.Module):
    """
    input: ([Batch, Channels, Frequency, Time]) x2
    output: ([Batch, 1,128]) x2
    """
    def __init__(self, in_chan=1, kernel_size=(5.5), pool_size1=(4,2), pool_size2=(2,2), pool_size3=(2,2), pool_size4=(2,2), gru=64):
        super(Siamese_CRNN_4, self).__init__()
        self.conv1 = conv_block(in_chan, 96, ker_size=kernel_size, stride=1, pool_size=pool_size1)   
        self.conv2 = conv_block(96, 128, ker_size=kernel_size, stride=1, pool_size=pool_size2)
        self.conv3 = conv_block(128, 128, ker_size=kernel_size, stride=1, pool_size=pool_size3)
        self.conv4 = conv_block(128, 128, ker_size=kernel_size, stride=1, pool_size=pool_size4)
        self.rnn1 = nn.GRU(input_size=128, hidden_size=gru, num_layers=2, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        # CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)        # [B,C,H,W] with h=1
        x = x.squeeze(2)         # [B,C,W]
        x = x.permute(0, 2, 1)   # [B,W,C]  # batch first = True
        # RNN
        x, _ = self.rnn1(x)
        return x[:, -1, :]

    def forward(self, in_1, in_2):
        out_1 = self.forward_once(in_1)
        out_2 = self.forward_once(in_2)

        return out_1, out_2
    
class Siamese_CRNN_2(nn.Module):
    """
    input: ([Batch, Channels, Frequency, Time]) x2
    output: ([Batch, 1,128]) x2
    """
    def __init__(self, in_chan=1, kernel_size=(5,5), pool_size1=(4,2), pool_size2=(2,2), gru=64):
        super(Siamese_CRNN_2, self).__init__()
        self.conv1 = conv_block(in_chan, 96, ker_size=kernel_size, stride=1, pool_size=pool_size1)   
        self.conv2 = conv_block(96, 128, ker_size=kernel_size, stride=1, pool_size=pool_size2)
        self.rnn1 = nn.GRU(input_size=128, hidden_size=gru, num_layers=2, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        # CNN
        x = self.conv1(x)
        x = self.conv2(x) 
        x = x.squeeze(2)         # [B,C,W]
        x = x.permute(0, 2, 1)   # [B,W,C]  # batch first = True
        # RNN
        x, _ = self.rnn1(x)
        return x[:, -1, :]

    def forward(self, in_1, in_2):
        out_1 = self.forward_once(in_1)
        out_2 = self.forward_once(in_2)

        return out_1, out_2

class CRNN_CLASS(nn.Module):
    
    def __init__(self):
        super(CRNN_CLASS, self).__init__()
        
        self.bn = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128, 256)  # Primo layer denso
        self.fc2 = nn.Linear(256, 1)   # Secondo layer denso
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Passaggi attraverso i layer densi con attivazioni ReLU, tranne l'ultimo con sigmoide
        x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)          # Output finale prima della sigmoide
        x = self.sigmoid(x)      # Applica sigmoid per ottenere la probabilità di appartenenza alla classe 1
        return x
       
class CRNN_FULL(nn.Module):
    """
    input: [Batch, Channels, Frequency, Time]
    output: [Batch, 1]
    """
    def __init__(self, in_chan=1):
        super(CRNN_FULL, self).__init__()
        self.conv1 = conv_block(in_chan, 96, ker_size=(5, 5), stride=1, pool_size=(4, 2))
        self.conv2 = conv_block(96, 128, ker_size=(5, 5), stride=1, pool_size=(2, 2))
        self.conv3 = conv_block(128, 128, ker_size=(5, 5), stride=1, pool_size=(5, 2))
        self.rnn1 = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(128, 256)  # Primo layer denso
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)        # [B,C,H,W] with h=1
        x = x.squeeze(2)         # [B,C,W]
        x = x.permute(0, 2, 1)   # [B,W,C]  # batch first = True
        # RNN
        x, _ = self.rnn1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)          # Output finale prima della sigmoide
        x = self.sigmoid(x)      # Applica sigmoid per ottenere la probabilità di appartenenza alla classe 1
        return x
        
        return x
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN_2()
    model.to(device)
    # inp = torch.randn((128,1,64,519))
    # inp = inp.to(device)
    # out = model(inp)
    # print(f'out {out.shape}')
    summary(model,(1,8,519))