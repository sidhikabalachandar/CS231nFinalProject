import torch.nn as nn
import torch.nn.functional as F

class baseline(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.enc_lin1 = nn.Linear(
            in_features=input_size, out_features=4096
        )
        self.enc_bn1 = nn.BatchNorm1d(
            in_features=4096
        )
        self.enc_lin2 = nn.Linear(
            in_features=4096, out_features=2048
        )
        self.enc_bn2 = nn.BatchNorm1d(
            in_features=2048
        )
        self.enc_lin3 = nn.Linear(
            in_features=2048, out_features=1024
        )
        self.enc_bn3 = nn.BatchNorm1d(
            in_features=1024
        )
        self.enc_lin4 = nn.Linear(
            in_features=1024, out_features=128
        )
        self.enc_bn4 = nn.BatchNorm1d(
            in_features=128
        )
        self.enc_lin5 = nn.Linear(
            in_features=128, out_features=128
        )
        self.enc_bn5 = nn.BatchNorm1d(
            in_features=128
        )
        self.dec_lin6 = nn.Linear(
            in_features=128, out_features=128
        )
        self.dec_bn6 = nn.BatchNorm1d(
            in_features=128
        )
        self.dec_lin7 = nn.Linear(
            in_features=128, out_features=1024
        )
        self.dec_bn7 = nn.BatchNorm1d(
            in_features=1024
        )
        self.dec_lin8 = nn.Linear(
            in_features=1024, out_features=2048
        )
        self.dec_bn8 = nn.BatchNorm1d(
            in_features=2048
        )
        self.dec_lin9 = nn.Linear(
            in_features=2048, out_features=4096
        )
        self.dec_bn9 = nn.BatchNorm1d(
            in_features=4096
        )
        self.dec_lin10 = nn.Linear(
            in_features=4096, out_features=6144
        )

    def forward(self, features):
        enc_1 = self.enc_bn1(F.leaky_relu(self.enc_lin1(features)))
        enc_2 = self.enc_bn2(F.leaky_relu(self.enc_lin2(enc_1)))
        enc_3 = self.enc_bn3(F.leaky_relu(self.enc_lin3(enc_2)))
        enc_4 = self.enc_bn4(F.leaky_relu(self.enc_lin4(enc_3)))
        enc_5 = self.enc_bn5(F.leaky_relu(self.enc_lin5(enc_4)))
        dec_1 = self.enc_bn1(F.leaky_relu(self.enc_lin1(enc_5)))
        dec_2 = self.enc_bn2(F.leaky_relu(self.enc_lin2(dec_1)))
        dec_3 = self.enc_bn3(F.leaky_relu(self.enc_lin3(dec_2)))
        dec_4 = self.enc_bn4(F.leaky_relu(self.enc_lin4(dec_3)))
        dec_5 = self.enc_bn5(F.leaky_relu(self.enc_lin5(dec_4)))
        return dec_5