
# python calculate_CD.py -m saved_models/lgan_train_sofa/generator_499.pt -t splits/sofa/train.txt -y gan -a saved_models/pointnet_train_sofa/best_490.pt
# python calculate_CD.py -m saved_models/maf_train_sofa/MAF_499.pt -t splits/sofa/train.txt -y maf -a saved_models/pointnet_train_sofa/best_490.pt

import torch
import argparse
from PointCloudDataset import PointCloudDataset
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as chamfer

from AE_models.gan import *
from AE_models.maf import *

def getCD(criterion, pc_1, pc_2):

    # pc_1 & pc_2 are both of size (batch_size, num_pts, 3)
    
    dist1, dist2, _, _ = criterion(pc_1, pc_2)
    dist = torch.mean(torch.sum(dist1 + dist2, axis = 1))
    return dist

def get_gan_data(G, ae, batch_size, noise_size):
    g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
    fake_images = G(g_fake_seed)
    fake_images = decode(ae, fake_images)
    return fake_images
    
def get_flow_data(model, ae, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_images = model.sample(device, n=batch_size)
    fake_images = decode(ae, fake_images)
    return fake_images

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name',  required=True, help = 'Path to model (.pt).')
    parser.add_argument('-t', '--train_path', required=True,  help='Path to training .txt file.')
    parser.add_argument('-y', '--type', required=True,  help='Either "gan" or "maf".')
    parser.add_argument('-a', '--ae_name',  required=True, help = 'Path to ae model (.pt).')
    args = parser.parse_args()

    #Load Train Data
    batch_size = 256
    noise_size=128
    trainset = PointCloudDataset(path_to_data = args.train_path)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Load Model
    model = torch.load(args.model_name)
    model.eval()

    ae = load_ae(args.ae_name)

    if args.type == "gan":
        example_fake = get_gan_data(model, ae, batch_size, noise_size)
    elif args.type == "maf":
        example_fake = get_flow_data(model, ae, batch_size)

    for example_real, _ in trainloader: # get first batch of real examples
        example_real = example_real.type(dtype)
        example_real = torch.reshape(example_real, (-1, 2048, 3))
        break

    criterion = chamfer.chamfer_3DDist()
    
    CD = 0
    for i in range(batch_size):
        for j in range(batch_size):
            CD += getCD(criterion, example_fake[i], example_real[j])

    average_CD = CD/(i*j)
    print('Average CD: {}'.format(average_CD))

if __name__ == "__main__":
    main()