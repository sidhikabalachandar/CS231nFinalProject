import torch
import torch.nn as nn
import torch.optim as optim

import open3d as o3d

import numpy as np 
from AE import AE
from PointCloudDataset import PointCloudDataset



path_car_loss = "./car_loss.txt"
file_handle = open(path_car_loss, "a")

path_car_train = "data/shape_net_core_uniform_samples_2048/car_train.txt"
path_car_val   = "data/shape_net_core_uniform_samples_2048/car_val.txt"
path_car_test  = "data/shape_net_core_uniform_samples_2048/car_test.txt"





batch_size = 256
epochs = 500
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

    nn.Linear(6144, out_features=4096),
    nn.LeakyReLU(),
    nn.BatchNorm1d(4096),
    
    nn.Linear(4096, out_features=2048),
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
    
    nn.Linear(2048, out_features=4096),
    nn.LeakyReLU(),
    nn.BatchNorm1d(4096),

    nn.Linear(4096, out_features=6144),
)





#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)


# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# mean-squared error loss
criterion = nn.MSELoss()

cur_best_val_loss = 10 #really big number
cur_best_train_loss = 10 #really big number
cur_best_epoch = 0
cur_best_model = None
for epoch in range(epochs):

    train_loss = 0
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
        curr_train_loss = criterion(outputs, batch_features)

        # compute accumulated gradients
        curr_train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        train_loss += curr_train_loss.item()

    # compute the epoch training loss
    train_loss = train_loss / len(trainloader)

    with torch.no_grad():
        val_loss = 0
        for index, (batch_features, _) in enumerate(valloader):
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device

            batch_features = batch_features.to(device).float()

            # compute reconstructions
            outputs = model(batch_features)

            # compute training reconstruction loss
            curr_val_loss = criterion(outputs, batch_features)

            # add the mini-batch training loss to epoch loss
            val_loss += curr_val_loss.item()

        # compute the epoch training loss
        val_loss = val_loss / len(valloader)
    
    # display the epoch training and validatation loss
    epoch_str = "epoch : {}/{}, train loss = {:.4f}, val loss = {:.4f}".format(epoch + 1, epochs, train_loss, val_loss)
    print(epoch_str)
    file_handle.write(epoch_str)
    
    if(epoch % 100 == 0):
        torch.save(model, "./{}_car_model.pt".format(epoch))
    
    if(val_loss < cur_best_loss):
        cur_best_val_loss = val_loss
        cur_best_train_loss = train_loss
        cur_best_epoch = epoch + 1
        cur_best_model = model

        
#Save and print best model
torch.save(cur_best_model, './best_{}_car_model.pt'.format(cur_best_epoch))
file_handle.close()

best_epoch_str = "best model found on epoch : {}/{}, train loss = {:.4f}, val loss = {:.4f}".format(cur_best_epoch, epochs, cur_best_train_loss, cur_best_val_loss)
print(best_epoch_str)