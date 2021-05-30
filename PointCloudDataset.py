import os
import torch
import random
import numpy as np
import open3d as o3d

class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_data):
        
        files = []

        # open file and read the content in a list
        with open(path_to_data, 'r') as filehandle:
            filecontents = filehandle.readlines()

            for line in filecontents:
                # remove linebreak which is the last character of the string
                current_place = line[:-1]

                # add item to the list
                files.append(current_place)
                    
        data_size = len(files)
        
        self.data_size = data_size
        self.files = random.sample(files, self.data_size)
        
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        
        point_file_name = self.files[idx]
        point_cloud = o3d.io.read_point_cloud(point_file_name)
        point_cloud_in_numpy = np.asarray(point_cloud.points)
        
        # point_cloud_in_numpy of shape (2048, 3) ---> (2048*3)
        # point_file_name = "ramdom_characters.ply"
        return point_cloud_in_numpy.reshape(-1), point_file_name