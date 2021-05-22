import os
import torch
import numpy as np
import open3d as o3d

##### Global States #####
#Load model
model = torch.load('./best_496_car_model.pt')
model.eval()

#Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_interpolate(x, y, N):
    t_space = np.linspace(x, y, N)
    return t_space

def extract_latent_representation(folder_name, path_pc_1, path_pc_2):
    
    # Create folder
    if(!os.path.exists(folder_name)):
        os.mkdir(folder_name)
    
    # Load point cloud
    pc_1 = o3d.io.read_point_cloud(path_pc_1)
    pc_1_numpy = np.asarray(pc_1.points).reshape(-1)
    pc_2 = o3d.io.read_point_cloud(path_pc_2)
    pc_2_numpy = np.asarray(pc_2.points).reshape(-1)
    
    #Convert point cloud to tensor
    torch_pc_1 = torch.from_numpy(pc_1)
    torch_pc_2 = torch.from_numpy(pc_2)
    
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
    t_space = linear_interpolate(torch_pc_1, torch_pc_1, Points)
    
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
        o3d.io.write_point_cloud(".{}/time_{}.ply".format(folder_name, index), pcd_output)
    
if __name__ == "__main__":
    #Train - Train
    pc1_train_1 = "data/shape_net_core_uniform_samples_2048/02958343/abdbd823f44be240f69e2f67d9a307fc.ply"
    pc1_train_2 = "data/shape_net_core_uniform_samples_2048/02958343/892266d574578810afe717997470b28d.ply"
    extract_latent_representation("train_train", pc1_train_1, pc1_train_2)
    
    #Train - Test
    pc2_train_1 = "data/shape_net_core_uniform_samples_2048/02958343/abdbd823f44be240f69e2f67d9a307fc.ply"
    pc2_test_2 = "data/shape_net_core_uniform_samples_2048/02958343/627c561f2a45ac907c4509228487875f.ply"
    extract_latent_representation("train_test", pc2_train_1, pc2_test_2)
    
    #Test - Test
    pc3_test_1 = "data/shape_net_core_uniform_samples_2048/02958343/627c561f2a45ac907c4509228487875f.ply"
    pc3_test_2 = "data/shape_net_core_uniform_samples_2048/02958343/6dc347214e4ebc364512af8eced68fa8.ply"
    extract_latent_representation("test_test", pc3_test_1, pc3_test_2)