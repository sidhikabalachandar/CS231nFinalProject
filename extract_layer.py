import torch
import numpy as np
import open3d as o3d


def main():    
    
    #Load test point cloud
    path_to_car = "data/shape_net_core_uniform_samples_2048/02958343/627c561f2a45ac907c4509228487875f.ply"
    point_cloud = o3d.io.read_point_cloud(path_to_car)
    point_cloud_in_numpy = np.asarray(point_cloud.points)
    point_cloud_in_numpy = point_cloud_in_numpy.reshape(-1)
    
    #Convert point cloud to tensor
    input_tensor = torch.from_numpy(point_cloud_in_numpy)
    
    #find device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device).float()

    #load model
    model = torch.load('./best_496_car_model.pt')
    model.eval()
   
    input_tensor = torch.reshape(input_tensor, (1,input_tensor.size()[0]))

    #Print tensors
    x = input_tensor
    print(x.size())
    for l in list(model.children()):
        x = l(x)
        print(x.size())
    
    

if __name__ == "__main__":
    main()
