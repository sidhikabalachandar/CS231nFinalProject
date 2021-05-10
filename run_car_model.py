import torch
import torch.nn as nn
import torch.optim as optim

import open3d as o3d

import numpy as np 
from AE import AE
from PointCloudDataset import PointCloudDataset





path_car_train = "data/shape_net_core_uniform_samples_2048/car_train.txt"
path_car_val   = "data/shape_net_core_uniform_samples_2048/car_val.txt"
path_car_test  = "data/shape_net_core_uniform_samples_2048/car_test.txt"





batch_size = 10
epochs = 100
learning_rate = 1e-3






#Load Train, Val, Test Data
trainset = PointCloudDataset(path_to_data = path_car_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

valset = PointCloudDataset(path_to_data = path_car_val)
valloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = PointCloudDataset(path_to_data = path_car_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=24, shuffle=True, num_workers=2)






# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu

model = nn.Sequential(

    nn.Linear(6144, out_features=2048),
    nn.LeakyReLU(),
    nn.BatchNorm1d(2048),
    
    nn.Linear(2048, out_features=1024),
    nn.LeakyReLU(),
    nn.BatchNorm1d(1024),
    
    nn.Linear(1024, out_features=128),
    nn.LeakyReLU(),
    nn.BatchNorm1d(128),
    
    nn.Linear(128, out_features=128),
    nn.LeakyReLU(),
    nn.BatchNorm1d(128),
    
    nn.Linear(128, out_features=128),
    nn.LeakyReLU(),
    nn.BatchNorm1d(128),
   
    nn.Linear(128, out_features=1024),
    nn.LeakyReLU(),
    nn.BatchNorm1d(1024),
    
    nn.Linear(1024, out_features=2048),
    nn.LeakyReLU(),
    nn.BatchNorm1d(2048),
    
    nn.Linear(2048, out_features=6144),
)






#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# mean-squared error loss
criterion = nn.MSELoss()



for epoch in range(epochs):

    loss = 0
    for index, (batch_features, _) in enumerate(trainloader):
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device

        batch_features = batch_features.to(device).float()

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        outputs = model(batch_features)

        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features)

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(trainloader)

    # display the epoch training loss
    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))
