# Layout

AE_models folder -- folder with different AE models

**create_splits.py :**
- split data into train, val, and test
- inputs: 
	- file with list of human readable objects to include in data, name
- outputs: 
	- splits/name/file of list of paths to training examples
	- splits/name/file of list of paths to val examples
	- splits/name/file of list of paths to test examples

**train_model.py:**
- trains AE model
- inputs: 
	- path to file with train paths
	- path to file with val paths
	- name
- outputs: 
	- saved_models/name/models saved every 100 epochs
	- saved_models/name/models saved after best epoch
                              
**test_model.py** 
- tests AE model
- inputs: 
	- path to file with test paths
	- name
- outputs: 
	- predicted/name/ply files for test examples
               
**interpolate_layer.py** 
- interpolation
- inputs: 
	- path to folder with decoded training examples
	- path to folder with decoded testing examples
	- model
	- output_prefix
-  outputs: 
	- interpolation/output_prefix/train_train folder for interpolated decoded output
	- interpolation/output_prefix/train_test folder for interpolated decoded output
	- interpolation/output_prefix/test_test folder for interpolated decoded output
               
**PointCloudDataset.py** 
- input to dataloader
- inputs: 
- outputs: 


# Progress:
# Starting from scratch, building autoencoder


# Quick Start

**Get data:**

`./download_data.sh`

**Create test, val, train splits:**
```
python create_splits.py -c guitar bus \
			-n guitar_bus
```
- Creates folder 
	- splits/guitar_bus
- Creates files 
	- splits/guitar_bus/train.txt
	- splits/guitar_bus/val.txt
	- splits/guitar_bus/test.txt

**Train model:**
```
python train_model.py -t splits/guitar_bus/train.txt \
		      -v splits/guitar_bus/val.txt \
		      -n train_guitar_bus
```
- Creates folder 
	- saved_models/train_guitar_bus
- Creates files 
	- saved_models/train_guitar_bus/losses.txt
	- saved_models/train_guitar_bus/\*.pt
	- saved_models/train_guitar_bus/best_\*.pt 

**Test model:**
```
python test_model.py -t splits/guitar_bus/test.txt \
		     -m saved_models/train_guitar_bus/best_5.pt \
		     -n test_guitar_bus
```
- Creates folder 
	- saved_models/predicted
- Creates files 
	- saved_models/predicted/\*.ply


**Test model:**
```
python interpolate_layer.py -pc1 shape_net_core_uniform_samples_2048/03467517/3d65570b55933c86ec7e7dbd8120b3cb.ply \
			    -pc2 shape_net_core_uniform_samples_2048/03467517/10b65379796e96091c86d29189611a06.ply \
			    -m saved_models/train_guitar_bus/best_5.pt \
			    -n interpolate_guitar_bus
```
- Creates folders 
	- interpolation/output_prefix/train_train
	- interpolation/output_prefix/train_test
	- interpolation/output_prefix/test_test
- Creates files
	- interpolation/output_prefix/\*/time_\*.ply
