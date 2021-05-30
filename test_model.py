import os
import torch
import open3d as o3d
from PointCloudDataset import PointCloudDataset
import argparse


# Globals
predicted_folder = "predicted"

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_path', required=True, help='Path to testing .txt file.')
    parser.add_argument('-m', '--model_path', required=True, help='Path to model')
    parser.add_argument('-n', '--folder_name', required=True,
                        help='Name of folder to save loss(.txt) and model(.pt) in.')
    args = parser.parse_args()

    folder_name = args.folder_name;
    testset = PointCloudDataset(path_to_data=args.test_path)
    testloader = torch.utils.data.DataLoader(testset, batch_size=24, shuffle=True, num_workers=2)

    model = torch.load(args.model_path)
    model.eval()

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create folder
    os.makedirs(os.path.join(predicted_folder, folder_name), exist_ok=True)

    # for predictions
    for single_test_feature, single_test_path in testloader:
        input_tensor = single_test_feature.to(device).float()  # (128, 6144)

        print("input_path:{}".format(single_test_path[0]))

        output_tensor = model(input_tensor)  # (128, 6144,)
        reshape_output_tensor = output_tensor[0].reshape(2048, 3)  # (2048, 3)

        pcd_output = o3d.geometry.PointCloud()
        pcd_output.points = o3d.utility.Vector3dVector(reshape_output_tensor.detach().cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(predicted_folder, folder_name, "predict_ae.ply"), pcd_output)


        reshape_input_tensor = input_tensor[0].reshape(2048, 3)  # (2048, 3)

        pcd_input = o3d.geometry.PointCloud()
        pcd_input.points = o3d.utility.Vector3dVector(reshape_input_tensor.detach().cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(predicted_folder, folder_name, "original_ae.ply"), pcd_input)
        break


if __name__ == "__main__":
    main()
