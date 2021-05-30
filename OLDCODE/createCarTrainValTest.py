import os

path_to_car = "data/shape_net_core_uniform_samples_2048/02958343"

split_train = .90
split_val   = .09
split_test  = .01

path_to_train = "data/shape_net_core_uniform_samples_2048/car_train.txt"
path_to_val   = "data/shape_net_core_uniform_samples_2048/car_val.txt"
path_to_test  = "data/shape_net_core_uniform_samples_2048/car_test.txt"

train_files = []
val_files   = []
test_files  = []

# for subdir in os.listdir(path_to_car):

subdir_path = os.listdir(os.path.join(path_to_car))
subdir_path = [f for f in subdir_path if f[-3:] == 'ply']

num_shapes  = len(subdir_path)
count_train = int(num_shapes*split_train)
count_val   = int(num_shapes*split_val)
count_test  = num_shapes - count_train - count_val

train = []
val = []
test = []


for file in subdir_path:
    file = os.path.join(path_to_car, file) + '\n'
    if len(train) <= count_train:
        train.append(file)
    elif len(val) <= count_val:
        val.append(file)
    else:
        test.append(file)

train_files.extend(train)
val_files.extend(val)
test_files.extend(test)
                      
with open(path_to_train, 'w') as train_file:
    train_file.writelines(train_files)
    
with open(path_to_val, 'w') as val_file:
    val_file.writelines(val_files)
    
with open(path_to_test, 'w') as test_file:
    test_file.writelines(test_files)

len_total_files = len(train_files) + len(val_files) + len(test_files)
p_train = len(train_files)/len_total_files
p_val   = len(val_files)/len_total_files
p_test  = len(test_files)/len_total_files

print("p_train:{}, p_val:{}, p_test:{}".format(p_train, p_val, p_test))






