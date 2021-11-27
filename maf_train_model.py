import os
import torch
# import numpy as np
import torch.optim as optim
from PointCloudDataset import PointCloudDataset
import argparse
from AE_models.maf import *

# Get Chamfer Distance
# import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as chamfer

# Global
saved_models = "saved_models"


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_path', required=True, help='Path to training .txt file.')
    parser.add_argument('-m', '--model_name',  required=False, default = None, help = 'Path to model (.pt).')
    parser.add_argument('-n', '--folder_name', required=True,
                        help='Name of folder to save loss(.txt) and model(.pt) in.')
    args = parser.parse_args()

    folder_name = args.folder_name
    os.makedirs(os.path.join(saved_models, folder_name), exist_ok=True)
    path_loss = os.path.join(saved_models, folder_name, 'losses.txt')

    batch_size = 128
    epochs = 500
    learning_rate = 1e-3

    encoder_name = args.model_name

    # Load Train, Val, Test Data
    trainset = PointCloudDataset(path_to_data=args.train_path)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    maf_model = MAF(
        input_size=2, hidden_size=args.hidden_size, n_hidden=1, n_flows=args.n_flows
    ).to(device)
    maf_optimizer = torch.optim.Adam(maf_model.parameters(), lr=1e-3, weight_decay=1e-6)


    run_a_maf(maf_model, maf_optimizer, trainloader, encoder_name, show_every=250,
              batch_size=batch_size, num_epochs=epochs, saved_models=saved_models, folder_name=folder_name,
              path_loss=path_loss)


if __name__ == "__main__":
    main()
