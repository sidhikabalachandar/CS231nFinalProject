import torch
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(self, zdim, input_dim=3, use_deterministic_encoder=False):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.r1 = nn.ReLU()

        self.conv2 = nn.Conv1d(128, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.r2 = nn.ReLU()

        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)
        self.r3 = nn.ReLU()

        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 256)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc_r1 = nn.ReLU()

        self.fc2 = nn.Linear(256, 128)
        self.fc_bn2 = nn.BatchNorm1d(128)
        self.fc_r2 = nn.ReLU()

        self.fc3 = nn.Linear(128, zdim)

        self.dec_lin1 = nn.Linear(
            in_features=128, out_features=128
        )
        self.lr1 = nn.LeakyReLU()
        self.dec_bn1 = nn.BatchNorm1d(
            num_features=128
        )
        self.dec_lin2 = nn.Linear(
            in_features=128, out_features=1024
        )
        self.lr2 = nn.LeakyReLU()
        self.dec_bn2 = nn.BatchNorm1d(
            num_features=1024
        )
        self.dec_lin3 = nn.Linear(
            in_features=1024, out_features=2048
        )
        self.lr3 = nn.LeakyReLU()
        self.dec_bn3 = nn.BatchNorm1d(
            num_features=2048
        )
        self.dec_lin4 = nn.Linear(
            in_features=2048, out_features=4096
        )
        self.lr4 = nn.LeakyReLU()
        self.dec_bn4 = nn.BatchNorm1d(
            num_features=4096
        )
        self.dec_lin5 = nn.Linear(
            in_features=4096, out_features=6144
        )

    def forward(self, x):
        # encoder
        x = x.transpose(1, 2)
        x = self.r1(self.bn1(self.conv1(x)))
        x = self.r2(self.bn2(self.conv2(x)))
        x = self.r3(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)
        ms = self.fc_r1(self.fc_bn1(self.fc1(x)))
        ms = self.fc_r2(self.fc_bn2(self.fc2(ms)))
        ms = self.fc3(ms)

        #decoder
        dec_1 = self.dec_bn6(self.lr6(self.dec_lin6(ms)))
        dec_2 = self.dec_bn7(self.lr7(self.dec_lin7(dec_1)))
        dec_3 = self.dec_bn8(self.lr8(self.dec_lin8(dec_2)))
        dec_4 = self.dec_bn9(self.lr9(self.dec_lin9(dec_3)))
        dec_5 = self.dec_lin10(dec_4)
        return dec_5