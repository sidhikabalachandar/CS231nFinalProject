import os
import random
import argparse


# Globals
split_folder = "splits"
data_path    = "shape_net_core_uniform_samples_2048"

split_train  = .90
split_val    = .09
split_test   = .01

snc_synth_id_to_category = {
    '02691156': 'airplane',  '02773838': 'bag',        '02801938': 'basket',
    '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
    '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
    '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
    '02954340': 'cap',       '02958343': 'car',        '03001627': 'chair',
    '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
    '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
    '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
    '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
    '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone'
}


def snc_category_to_synth_id():
    inv_map = {v: k for k, v in snc_synth_id_to_category.items()}
    return inv_map


def main():
    '''
    usage: create_splits.py [-h] -c CATEGORY [CATEGORY ...] -n FOLDER_NAME

    optional arguments:
      -h, --help            show this help message and exit
      -c CATEGORY [CATEGORY ...], --category CATEGORY [CATEGORY ...]
                            List of categories.
      -n FOLDER_NAME, --folder_name FOLDER_NAME
                            Name of solder to save splits in
    '''
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', nargs='+', default=[], required=True, help='List of categories.')
    parser.add_argument('-n', '--folder_name', required=True, help='Name of folder to save splits in.')
    args = parser.parse_args()
    
    folder_name = args.folder_name
    
    # Map category to synth_id
    category_to_synth_id = snc_category_to_synth_id()

    # Get paths to data (ex: shape_net_core_uniform_samples_2048/02958343)
    paths_to_input_category =[]
    
    for object_type in args.category:
        synth_id = category_to_synth_id[object_type]
        paths_to_input_category.append(os.path.join(data_path, synth_id))

    # Get paths to train/val/test folders (ex: splits/name/train.txt)
    path_to_train = os.path.join(split_folder, folder_name, "train.txt")
    path_to_val   = os.path.join(split_folder, folder_name, "val.txt")
    path_to_test  = os.path.join(split_folder, folder_name, "test.txt")
    
    train_files, val_files, test_files = [], [], []
    
    # Loop over each category
    for path_to_category in paths_to_input_category:
        samples_of_category =  os.listdir(os.path.join(path_to_category))
        samples_of_category = [f for f in samples_of_category if f[-3:] == 'ply']
        
        # Shuffle samples
        random.shuffle(samples_of_category)
        
        num_samples = len(samples_of_category)
        count_train = int(num_samples*split_train)
        count_val   = int(num_samples*split_val)
        count_test  = num_samples - count_train - count_val
    
        train, val, test = [], [], []

        # Adding files to train/val/test
        for sample in samples_of_category:
            sample_path = os.path.join(path_to_category, sample) + '\n'
            if len(train) <= count_train:
                train.append(sample_path)
            elif len(val) <= count_val:
                val.append(sample_path)
            else:
                test.append(sample_path)
        
        
        train_files.extend(train)
        val_files.extend(val)
        test_files.extend(test)
    

    # Save paths to train/val/test
    os.makedirs(os.path.join(split_folder, folder_name), exist_ok=True)
    with open(path_to_train, 'w') as train_file:
        train_file.writelines(train_files)

    with open(path_to_val, 'w') as val_file:
        val_file.writelines(val_files)

    with open(path_to_test, 'w') as test_file:
        test_file.writelines(test_files)

        
    # Calculate summary statistics
    cnt_train = len(train_files)
    cnt_val   = len(val_files)
    cnt_test  = len(test_files)

    print("""Splits saved in: \n \
           path_to_train:{} --- {} \n \
           path_to_val:{} --- {} \n \
           path_to_test:{} --- {} """.format(path_to_train, cnt_train, path_to_val, cnt_val, path_to_test, cnt_test))
    
    

if __name__ == "__main__":
    main()


