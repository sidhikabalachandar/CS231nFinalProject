# Based off of: https://github.com/stevenygd/PointFlow/blob/master/models/networks.py
import torch
import torch.nn as nn


class pointnet(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        
        # input  : (N, 3, 2048)
        # output : (N, 2048*3)
        
        # Encode
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.lr1 = nn.LeakyReLU()
        
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.lr2 = nn.LeakyReLU()
        
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)
        self.lr3 = nn.LeakyReLU()
        
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)

        self.take_max = nn.MaxPool1d(2048)  # Converting (N, 512, P) --> (N, 512, 1)  selects maximum channel for each point
        
        #Need to add a reshape layer
        self.flatten_me = nn.Flatten() # Converted (N, 512, 1) into (N, 512)
        
        self.fc1 = nn.Linear(512, 256)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.lr5 = nn.LeakyReLU()
        
        self.fc2 = nn.Linear(256, 128)
        self.fc_bn2 = nn.BatchNorm1d(128)
        self.lr6 = nn.LeakyReLU()
        
        self.fc3 = nn.Linear(128, 128)  # (N, 128)
        
        
        # Decode
        self.dec_lin7 = nn.Linear(128, 1024)
        self.dec_bn7 = nn.BatchNorm1d(1024)
        self.lr7 = nn.LeakyReLU()
        
        self.dec_lin8 = nn.Linear(in_features=1024, out_features=2048)
        self.dec_bn8 = nn.BatchNorm1d(num_features=2048)
        self.lr8 = nn.LeakyReLU()
        
        self.dec_lin9 = nn.Linear(in_features=2048, out_features=4096)
        self.dec_bn9 = nn.BatchNorm1d(num_features=4096)
        self.lr9 = nn.LeakyReLU()
        
        
        self.dec_lin10 = nn.Linear(in_features=4096, out_features=6144)
        

    def forward(self, x):
        # Encode
        x = self.lr1(self.bn1(self.conv1(x)))
        x = self.lr2(self.bn2(self.conv2(x)))
        x = self.lr3(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        
        x = self.take_max(x)
        x = self.flatten_me(x)

        assert x != None
        x = self.lr5(self.fc_bn1(self.fc1(x)))
        assert x != None
        x = self.lr6(self.fc_bn2(self.fc2(x)))
        assert x != None
        x = self.fc3(x)
        
        assert x != None
        # Decoder
        x = self.lr7(self.dec_bn7(self.dec_lin7(x)))
        assert x != None
        x = self.lr8(self.dec_bn8(self.dec_lin8(x)))
        assert x != None
        x = self.lr9(self.dec_bn9(self.dec_lin9(x)))
        assert x != None
        
        x = self.dec_lin10(x)
        assert x != None
        
