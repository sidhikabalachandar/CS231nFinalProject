import os
import torch
import argparse
import numpy as np
import open3d as o3d

# ##### Global States #####

interpolation_folder = "interpolation"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_interpolate(x, y, N):
    t_space = np.linspace(x.detach().cpu().numpy(), y.detach().cpu().numpy(), N)
    for i in range(N):
        print(t_space[i])
    t_space = torch.from_numpy(t_space).to(device).float()
    return t_space

def extract_latent_representation():
    '''
    usage: interpolate_layer.py [-h] -p1 PATH_PC_1 -p2 PATH_PC_2 -m MODEL_NAME -n FOLDER_NAME

    optional arguments:
      -h, --help            show this help message and exit
      -pc1 PATH_PC_1, --path_pc_1 PATH_PC_1
                            Path to point cloud 1 as .ply.
      -pc2 PATH_PC_2, --path_pc_2 PATH_PC_2
                            Path to point cloud 1 as .ply.
      -m MODEL_NAME, --model_name MODEL_NAME
                            Path to model (.pt).
      -n FOLDER_NAME, --folder_name FOLDER_NAME
                            Name of folder to save interpolation in.
    '''
    
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-pc1', '--path_pc_1', required=True, help = 'Path to point cloud 1 as .ply.')
    parser.add_argument('-pc2', '--path_pc_2', required=True, help = 'Path to point cloud 1 as .ply.')
    parser.add_argument('-m', '--model_name',  required=True, help = 'Path to model (.pt).')
    parser.add_argument('-n', '--folder_name',  required=True, help = 'Name of folder to save interpolation in.')
    args = parser.parse_args()

    # Load Model
    model = torch.load(args.model_name)
    model.eval()
    
    folder_name, path_pc_1, path_pc_2 = args.folder_name, args.path_pc_1, args.path_pc_2
    
    print("Writing to folder:{}".format(folder_name))

    # Create folder
    os.makedirs(os.path.join(interpolation_folder, folder_name), exist_ok=True)
    
    # Load point cloud
    pc_1 = o3d.io.read_point_cloud(path_pc_1)
    pc_1_numpy = np.asarray(pc_1.points).reshape(-1)
    pc_2 = o3d.io.read_point_cloud(path_pc_2)
    pc_2_numpy = np.asarray(pc_2.points).reshape(-1)
    
    #Convert point cloud to tensor
    torch_pc_1 = torch.from_numpy(pc_1_numpy)
    torch_pc_2 = torch.from_numpy(pc_2_numpy)
    
    #Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_pc_1 = torch_pc_1.to(device).float()
    torch_pc_2 = torch_pc_2.to(device).float()
    
    #Reshape tensor (required because we expect batches)
    torch_pc_1 = torch.reshape(torch_pc_1, (1,torch_pc_1.size()[0]))
    torch_pc_2 = torch.reshape(torch_pc_2, (1,torch_pc_2.size()[0]))
    
    #Extract Latent 128 vector ONLY up to 14 layers
    for layer in list(model.children())[:14]:
        torch_pc_1 = layer(torch_pc_1)
        torch_pc_2 = layer(torch_pc_2)
    

    #Linearly interpolate between 10 points
    Points = 10
    t_space = linear_interpolate(torch_pc_1, torch_pc_2, Points)
    
    #Convert latent representation to full point cloud and save 
    for index, t in enumerate(t_space):
        
        torch_intemediate_pc = t
        #Run decoder model 
        for layer in list(model.children())[14:]:
            torch_intemediate_pc  = layer(torch_intemediate_pc)
        
        
        #Save point cloud as ply files in folder_name
        torch_intemediate_pc = torch_intemediate_pc.reshape(2048, 3)
        pcd_output = o3d.geometry.PointCloud()
        pcd_output.points = o3d.utility.Vector3dVector(torch_intemediate_pc.detach().cpu().numpy())
        #o3d.io.write_point_cloud("{}/time_{}.ply".format(folder_name, index), pcd_output)
        o3d.io.write_point_cloud(os.path.join(interpolation_folder, folder_name, "time_{}.ply".format(index)), pcd_output)
        
        
    file = 'shape_net_core_uniform_samples_2048/02958343/2c6f09768e487fc792bf77570fd2f158.ply'
    file_pc = o3d.io.read_point_cloud(file)
    file_np = np.asarray(file_pc.points).reshape(-1)
    file_torch = torch.from_numpy(file_np)
    file_torch = file_torch.to(device).float()
    file_torch = torch.reshape(file_torch, (1,file_torch.size()[0]))
    for layer in list(model.children())[:14]:
        file_torch = layer(file_torch)
    print(file_torch)

if __name__ == "__main__":
    extract_latent_representation()
