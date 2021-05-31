import torch.nn as nn
import torch.nn.functional as F

class baseline(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.enc_lin1 = nn.Linear(
            in_features=input_size, out_features=4096
        )
        self.lr1 = nn.LeakyReLU()
        self.enc_bn1 = nn.BatchNorm1d(
            num_features=4096
        )
        self.enc_lin2 = nn.Linear(
            in_features=4096, out_features=2048
        )
        self.lr2 = nn.LeakyReLU()
        self.enc_bn2 = nn.BatchNorm1d(
            num_features=2048
        )
        self.enc_lin3 = nn.Linear(
            in_features=2048, out_features=1024
        )
        self.lr3 = nn.LeakyReLU()
        self.enc_bn3 = nn.BatchNorm1d(
            num_features=1024
        )
        self.enc_lin4 = nn.Linear(
            in_features=1024, out_features=128
        )
        self.lr4 = nn.LeakyReLU()
        self.enc_bn4 = nn.BatchNorm1d(
            num_features=128
        )
        self.enc_lin5 = nn.Linear(
            in_features=128, out_features=128
        )
        self.lr5 = nn.LeakyReLU()
        self.enc_bn5 = nn.BatchNorm1d(
            num_features=128
        )
        self.dec_lin6 = nn.Linear(
            in_features=128, out_features=128
        )
        self.lr6 = nn.LeakyReLU()
        self.dec_bn6 = nn.BatchNorm1d(
            num_features=128
        )
        self.dec_lin7 = nn.Linear(
            in_features=128, out_features=1024
        )
        self.lr7 = nn.LeakyReLU()
        self.dec_bn7 = nn.BatchNorm1d(
            num_features=1024
        )
        self.dec_lin8 = nn.Linear(
            in_features=1024, out_features=2048
        )
        self.lr8 = nn.LeakyReLU()
        self.dec_bn8 = nn.BatchNorm1d(
            num_features=2048
        )
        self.dec_lin9 = nn.Linear(
            in_features=2048, out_features=4096
        )
        self.lr9 = nn.LeakyReLU()
        self.dec_bn9 = nn.BatchNorm1d(
            num_features=4096
        )
        self.dec_lin10 = nn.Linear(
            in_features=4096, out_features=6144
        )

    def forward(self, features):
        enc_1 = self.enc_bn1(self.lr1(self.enc_lin1(features)))
        enc_2 = self.enc_bn2(self.lr2(self.enc_lin2(enc_1)))
        enc_3 = self.enc_bn3(self.lr3(self.enc_lin3(enc_2)))
        enc_4 = self.enc_bn4(self.lr4(self.enc_lin4(enc_3)))
        enc_5 = self.enc_bn5(self.lr5u(self.enc_lin5(enc_4)))
        dec_6 = self.dec_bn6(self.lr6(self.dec_lin6(enc_5)))
        dec_7 = self.dec_bn7(self.lr7(self.dec_lin7(dec_6)))
        dec_8 = self.dec_bn8(self.lr8(self.dec_lin8(dec_7)))
        dec_9 = self.dec_bn9(self.lr9(self.dec_lin9(dec_8)))
        dec_10 = self.dec_lin10(dec_9)
        return dec_10