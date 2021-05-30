# Quick Start
`./download_data.sh`

AE_models folder -- folder with different AE models

create_splits.py -- split data into train, val, and test
               -- -- previously createCarTrainValTest.py
               -- -- inputs: file with list of human readable objects to include in data, name
               -- -- outputs: splits/name/file of list of paths to training examples
                              splits/name/file of list of paths to val examples
                              splits/name/file of list of paths to test examples

train_model.py -- trains AE model
               -- -- previously run_car_model.py
               -- -- inputs: path to file with train paths
                             path to file with val paths
                             name
               -- -- outputs: saved_models/name/models saved every 100 epochs
                              saved_models/name/models saved after best epoch
                              
test_model.py -- tests AE model
               -- -- previously test_car_model.py
               -- -- inputs: path to file with test paths
                             name
               -- -- outputs: predicted/name/ply files for test examples
               
interpolate_layer.py -- interpolation
               -- -- previously extract_layer.py
               -- -- inputs: path to folder with decoded training examples
                             path to folder with decoded testing examples
                             model
                             output_prefix
               -- -- outputs: interpolation/output_prefix/train_train folder for interpolated decoded output
                              interpolation/output_prefix/train_test folder for interpolated decoded output
                              interpolation/output_prefix/test_test folder for interpolated decoded output
               
PointCloudDataset.py -- input to dataloader
               -- -- inputs: 
               -- -- outputs: 


# Progress:
# Starting from scratch, building autoencoder