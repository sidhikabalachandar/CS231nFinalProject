import os
import torch
# import numpy as np
import torch.optim as optim
from PointCloudDataset import PointCloudDataset
import argparse
from AE_models.gan import *

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
    # path_loss = os.path.join(saved_models, folder_name, 'losses.txt')

    # file_handle = open(path_loss, "a")
    batch_size = 256
    epochs = 500
    learning_rate = 1e-3
    # num_points = 2048

    # Load Train, Val, Test Data
    trainset = PointCloudDataset(path_to_data=args.train_path)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # valset = PointCloudDataset(path_to_data=args.val_path)
    # valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    D = discriminator()
    G = generator()

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D.to(device)
    G.to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    D_solver = optim.Adam(D.parameters(), lr=learning_rate)
    G_solver = optim.Adam(G.parameters(), lr=learning_rate)

    run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, trainloader, show_every=250,
              batch_size=batch_size, noise_size=128, num_epochs=epochs)


if __name__ == "__main__":
    main()
