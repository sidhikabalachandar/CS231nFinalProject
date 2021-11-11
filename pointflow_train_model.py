import os
import torch
import numpy as np
import torch.optim as optim
from PointCloudDataset import PointCloudDataset
import argparse
from AE_models.pointflow import PointFlow

# Get Chamfer Distance
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as chamfer

# Global
saved_models = "saved_models"


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_path', required=True, help='Path to training .txt file.')
    parser.add_argument('-v', '--val_path', required=True, help='Path to validation .txt file.')
    parser.add_argument('-n', '--folder_name', required=True,
                        help='Name of folder to save loss(.txt) and model(.pt) in.')
    args = parser.parse_args()

    folder_name = args.folder_name
    os.makedirs(os.path.join(saved_models, folder_name), exist_ok=True)
    path_loss = os.path.join(saved_models, folder_name, 'losses.txt')

    file_handle = open(path_loss, "a")
    batch_size = 256
    epochs = 500
    learning_rate = 1e-3
    num_points = 2048

    # Load Train, Val, Test Data
    trainset = PointCloudDataset(path_to_data=args.train_path)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    valset = PointCloudDataset(path_to_data=args.val_path)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = PointFlow(num_points * 3)

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Trying Chamfer Distance
    #     # mean-squared error loss
    #     criterion = nn.MSELoss()
    criterion = chamfer.chamfer_3DDist()

    cur_best_val_loss = np.inf  # really big number
    cur_best_train_loss = np.inf  # really big number
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

            #             # compute training reconstruction loss
            #             curr_train_loss = criterion(outputs, batch_features)

            #             # compute accumulated gradients
            #             curr_train_loss.backward()

            dist1, dist2, _, _ = criterion(torch.reshape(outputs, (-1, num_points, 3)),
                                           torch.reshape(batch_features, (-1, num_points, 3)))

            curr_train_loss = torch.mean(torch.sum(dist1 + dist2, axis=1))
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

                #                 # compute reconstructions
                val_outputs = model(batch_features)

                #                 # compute training reconstruction loss
                #                 curr_val_loss = criterion(outputs, batch_features)

                dist1, dist2, _, _ = criterion(torch.reshape(val_outputs, (-1, num_points, 3)),
                                               torch.reshape(batch_features, (-1, num_points, 3)))

                curr_val_loss = torch.mean(torch.sum(dist1 + dist2, axis=1))

                # add the mini-batch training loss to epoch loss
                val_loss += curr_val_loss.item()

            # compute the epoch training loss
            val_loss = val_loss / len(valloader)

        # display the epoch training and validatation loss
        epoch_str = "epoch : {}/{}, train loss = {:.4f}, val loss = {:.4f}".format(epoch + 1, epochs, train_loss,
                                                                                   val_loss)
        print(epoch_str)
        file_handle.write(epoch_str + "\n")

        if ((epoch + 1) % 100 == 0):
            torch.save(model, "{}/{}.pt".format(os.path.join(saved_models, folder_name), epoch))

        if (val_loss < cur_best_val_loss):
            cur_best_val_loss = val_loss
            cur_best_train_loss = train_loss
            cur_best_epoch = epoch + 1
            cur_best_model = model

    # Save and print best model
    torch.save(cur_best_model, '{}/best_{}.pt'.format(os.path.join(saved_models, folder_name), cur_best_epoch))
    file_handle.close()
    best_epoch_str = "best model found on epoch : {}/{}, train loss = {:.4f}, val loss = {:.4f}".format(cur_best_epoch,
                                                                                                        epochs,
                                                                                                        cur_best_train_loss,
                                                                                                        cur_best_val_loss)
    print(best_epoch_str)


if __name__ == "__main__":
    main()