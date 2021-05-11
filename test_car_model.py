import torch
import torch.nn as nn
import torch.optim as optim

import open3d as o3d

import numpy as np 
from AE import AE
from PointCloudDataset import PointCloudDataset


path_car_test  = "data/shape_net_core_uniform_samples_2048/car_test.txt"

testset = PointCloudDataset(path_to_data = path_car_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=24, shuffle=True, num_workers=2)


model = torch.load('./car_model.pt')
model.eval()


#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#for predictions
name = "car_test_pcloud"

for single_test_feature, single_test_path in testloader:
    input_tensor = single_test_feature.to(device).float()         #(128, 6144)
    
    print("input_path:{}".format(single_test_path[0]))
    
    output_tensor = model(input_tensor)                #(128, 6144,)
    reshape_output_tensor = output_tensor[0].reshape(2048, 3) #(2048, 3)
    
    pcd_output = o3d.geometry.PointCloud()
    pcd_output.points = o3d.utility.Vector3dVector(reshape_output_tensor.detach().cpu().numpy())
    o3d.io.write_point_cloud("./{}_predict_ae.ply".format(name), pcd_output)

    

    reshape_input_tensor = input_tensor[0].reshape(2048, 3) #(2048, 3)

    pcd_input = o3d.geometry.PointCloud()
    pcd_input.points = o3d.utility.Vector3dVector(reshape_input_tensor.detach().cpu().numpy())
    o3d.io.write_point_cloud("./{}_original_ae.ply".format(name), pcd_input)   

    
    break
