import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential

class MyLstm(nn.Module):
    def __init__(self,channel_num=30):
        super(MyLstm, self).__init__()
        # self.lstm = nn.LSTM(30,20,2,batch_first=True)
        # self.smooth = nn.Sequential(
        #     nn.ReLU(),
        #     nn.BatchNorm1d(20),
        # )
        self.lstm1 = nn.LSTM(channel_num,10,1,batch_first=True,bidirectional=True)
        self.lstm2 = nn.LSTM(20,5,1,batch_first=True,bidirectional=True)
        self.smooth = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(20),
        )
    def forward(self, x):
        # x = torch.transpose(x,1,2)
        # out,_ = self.lstm(x)
        # out = self.smooth(out[:,-1,:])
        # return out

        x = torch.transpose(x,1,2)
        out,_ = self.lstm1(x)
        out,_ = self.lstm2(out)
        return out[:,-1,:]


class EEGNet(nn.Module):
    def __init__(self, channel_num=30):
        super(EEGNet, self).__init__()
        self.drop_out = 0.25
        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=channel_num,  # input shape (1, C, T)
                out_channels=16,  # num_filters
                kernel_size=(1, 64),  # filter size
                bias=False
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(16)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # input shape (8, C, T)
                out_channels=32,  # num_filters
                kernel_size=(1, 22),  # filter size
                groups=16,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(32),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=32,  # input shape (16, 1, T//4)
                out_channels=32,  # num_filters
                kernel_size=(1, 16),  # filter size
                groups=32,
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.Conv2d(
                in_channels=32,  # input shape (16, 1, T//4)
                out_channels=32,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(32),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.block_4 = nn.Sequential(
            nn.Linear(32*11, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )

    def forward(self, x):
        #add dim
        x = x.unsqueeze(2)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.reshape(x.size(0), -1)
        x = self.block_4(x)
        return x


class Feature(nn.Module):
    def __init__(self, channel_num=30):
        super(Feature, self).__init__()
        self.eegnet = EEGNet(channel_num)
        self.lstm = MyLstm(channel_num)
        self.attention1 = Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.attention2 = Sequential(
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x1 = self.eegnet(x)  #32
        x2 = self.lstm(x)    #10
        x1 = x1 * self.attention1(x1)
        x2 = x2 * self.attention2(x2)
        return torch.cat([x1,x2],1)
        # return x1

class Predictor(nn.Module):
    def __init__(self, classes_num):
        super(Predictor, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(42, classes_num),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class EEGNetForCDA(nn.Module):
    def __init__(self, channel_num=30,classes_num=2):
        super(EEGNetForCDA, self).__init__()
        self.drop_out = 0.25
        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=channel_num,  # input shape (1, C, T)
                out_channels=16,  # num_filters
                kernel_size=(1, 64),  # filter size
                bias=False
            ),  # output shape (8, C, T)
            # nn.BatchNorm2d(16)  # output shape (8, C, T)
        )

        self.bn_1 = nn.Sequential(
            nn.BatchNorm2d(16)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # input shape (8, C, T)
                out_channels=32,  # num_filters
                kernel_size=(1, 22),  # filter size
                groups=16,
                bias=False
            ),  # output shape (16, 1, T)
            # nn.BatchNorm2d(32),  # output shape (16, 1, T)
        )

        self.bn_2 = nn.Sequential(
            nn.BatchNorm2d(32)  # output shape (8, C, T)
        )

        self.block_3 = nn.Sequential(
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.block_4 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=32,  # input shape (16, 1, T//4)
                out_channels=32,  # num_filters
                kernel_size=(1, 16),  # filter size
                groups=32,
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.Conv2d(
                in_channels=32,  # input shape (16, 1, T//4)
                out_channels=32,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            # nn.BatchNorm2d(32),  # output shape (16, 1, T//4)
        )

        self.bn_4 = nn.Sequential(
            nn.BatchNorm2d(32)  # output shape (8, C, T)
        )

        self.block_5 = nn.Sequential(
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.block_6 = nn.Sequential(
            nn.Linear(32*11, 128),
            nn.ReLU(),
            # nn.BatchNorm1d(128),
        )

        self.bn_6 = nn.Sequential(
            nn.BatchNorm1d(128),  # output shape (8, C, T)
        )
        
        self.block_7 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            # nn.BatchNorm1d(32),
        )

        self.bn_7 = nn.Sequential(
            nn.BatchNorm1d(32),  # output shape (8, C, T)
        )

        self.block_8 = nn.Sequential(
            nn.Linear(32, classes_num),
            nn.ReLU(),
        )

    def forward(self, x):
        #add dim
        # x = x.unsqueeze(2)
        # x1 = self.bn_1(self.block_1(x))
        # x2 = self.bn_2(self.block_2(x1))
        # x4 = self.bn_4(self.block_4(self.block_3(x2)))
        # x = self.block_5(x4)
        # x = x.reshape(x.size(0), -1)
        # x6 = self.bn_6(self.block_6(x))
        # x7 = self.bn_7(self.block_7(x6))
        # x = self.block_8(x7)
        x = x.unsqueeze(2)
        x1 = self.block_1(x)
        x2 = self.block_2(self.bn_1(x1))
        x4 = self.block_4(self.block_3(self.bn_2(x2)))
        x = self.block_5(self.bn_4(x4))
        x = x.reshape(x.size(0), -1)
        x6 = self.block_6(x)
        x7 = self.block_7(self.bn_6(x6))
        feature = self.bn_7(x7)
        x = self.block_8(feature)
        return x, feature#, x1, x2, x4, x6, x7

class Model(nn.Module):
    def __init__(self, channel_num, classes_num):
        super(Model, self).__init__()
        self.feature = Feature(channel_num)
        self.predictor = Predictor(classes_num)
    def forward(self, x):
        x1 = self.feature(x)
        x = self.predictor(x1)
        return x,x1

class MixModel(nn.Module):
    def __init__(self, channel_num, classes_num):
        super(MixModel, self).__init__()
        self.feature = Feature(channel_num)
        self.predictor = Predictor(classes_num)
    def forward(self, x):
        fea = self.feature(x)
        x = self.predictor(fea)
        return x, fea

if __name__ == '__main__':

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    input = torch.randn(64,5,384).to(device)

    model = Model(5,2).to(device)

    out = model(input)

    print(out)