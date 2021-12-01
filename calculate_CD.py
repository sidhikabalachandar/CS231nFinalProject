
# python calculate_CD.py -g saved_models/lgan_train_sofa/generator_499.pt -f saved_models/maf_train_sofa/MAF_499.pt -t splits/sofa/train.txt -a saved_models/pointnet_train_sofa/best_490.pt

import torch
import argparse
from PointCloudDataset import PointCloudDataset
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as chamfer

from AE_models.gan import *
from AE_models.maf import *

def getCD(criterion, pc_1, pc_2):
    dist1, dist2, _, _ = criterion(pc_1, pc_2)
    dist = torch.mean(torch.sum(dist1 + dist2, axis = 1))
    return dist

def get_gan_data(G, ae, batch_size, noise_size, num_points):
    g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
    fake_images = G(g_fake_seed)
    fake_images = decode(ae, fake_images).reshape((batch_size, num_points, 3))
    return fake_images
    
def get_flow_data(model, ae, batch_size, num_points):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_images = model.sample(device, n=batch_size)
    fake_images = decode(ae, fake_images).reshape((batch_size, num_points, 3))
    return fake_images

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gan_model_name',  required=True, help = 'Path to model (.pt).')
    parser.add_argument('-f', '--flow_model_name', required=True, help='Path to model (.pt).')
    parser.add_argument('-t', '--train_path', required=True,  help='Path to training .txt file.')
    parser.add_argument('-a', '--ae_name',  required=True, help = 'Path to ae model (.pt).')
    args = parser.parse_args()

    #Load Train Data
    batch_size = 1000
    noise_size=128
    num_points = 2048
    trainset = PointCloudDataset(path_to_data = args.train_path)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Load Model
    gan_model = torch.load(args.gan_model_name)
    gan_model.eval()

    flow_model = torch.load(args.flow_model_name)
    flow_model.eval()

    ae = load_ae(args.ae_name)

    gan_example_fake = get_gan_data(gan_model, ae, batch_size, noise_size, num_points)
    flow_example_fake = get_flow_data(flow_model, ae, batch_size, num_points)
    for i, (example, _) in enumerate(trainloader): # get first batch of real examples
        if i == 0:
            example_real = example.type(dtype)
            example_real = torch.reshape(example_real, (-1, 2048, 3))
        if i == 1:
            data_example_fake = example.type(dtype)
            data_example_fake = torch.reshape(data_example_fake, (-1, 2048, 3))
            break

    criterion = chamfer.chamfer_3DDist()

    for i in range(num_points):
        fake = gan_example_fake[i, :, :].repeat(batch_size, -1, -1)
        print(fake)
        print(fake.size())
        assert (1 == 2)
        average_CD += getCD(criterion, fake, example_real)


    # average_CD = getCD(criterion, flow_example_fake, example_real)
    # average_CD = getCD(criterion, data_example_fake, example_real)
    print('Average CD: {}'.format(average_CD))

    for i in range(num_points):
        for j in range(num_points):
            fake = gan_example_fake[i, :, :].repeat(batch_size, -1, -1)
            real = example_real[j, :, :].repeat(batch_size, -1, -1)
            print(fake)
            average_CD = getCD(criterion, fake, example_real)

if __name__ == "__main__":
    main()