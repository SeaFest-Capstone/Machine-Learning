import os
import zipfile
import random
import shutil
import tensorflow as tf
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.setrecursionlimit(10000)

root_path = ''
source_path = os.path.join(root_path, 'Raw Data')
classification_data_source_path = os.path.join(source_path, 'Classification')
freshness_data_source_path = os.path.join(source_path, 'Freshness')
clean_dataset = True
SPLIT_SIZE = 0.7

def labels_classification_directory_txt():
    outputfile	= "classification_labels.txt"
    folder		= "Raw Data/Classification"	
    pathsep		= "\\"
    classification_labels = []

    try:
        for path,dirs,files in os.walk(folder):
            sep = path.split(pathsep)[len(path.split(pathsep))-1]
            print(sep)
            classification_labels.append(sep)

    except IndexError:
        pass


    classification_labels.pop(0)

    # SAVE TO TXT
    try: 
        with open(outputfile, "x") as txtfile:		
            for path,dirs,files in os.walk(folder):
                sep = path.split(pathsep)[len(path.split(pathsep))-1]
                print(sep)
                txtfile.write("%s\n" % sep)
        txtfile.close()

    except FileExistsError:
        os.remove(outputfile)
        with open(outputfile, "x") as txtfile:	
            for path,dirs,files in os.walk(folder):
                sep = path.split(pathsep)[len(path.split(pathsep))-1]
                print(sep)
                txtfile.write("%s\n" % sep)

        txtfile.close()
        pass   
    return classification_labels

def create_train_val_dir(root_path, classification_labels):
    classification_fish_paths = {}
    classification_fish_train_paths = {}
    classification_fish_val_paths = {}

    datasets_dir = os.path.join(root_path, 'Datasets')
    
    if clean_dataset==True and os.path.exists(datasets_dir)==True :
        shutil.rmtree(datasets_dir)

    freshness_fish_dir = os.path.join(datasets_dir, 'Freshness')

    freshness_fresh_fish_dir = os.path.join(freshness_fish_dir, 'Fresh')
    freshness_fresh_train_dir = os.path.join(freshness_fresh_fish_dir, 'train')
    freshness_fresh_val_dir = os.path.join(freshness_fresh_fish_dir, 'val') 

    freshness_non_fresh_fish_dir = os.path.join(freshness_fish_dir, 'Non Fresh')
    freshness_non_fresh_train_dir = os.path.join(freshness_non_fresh_fish_dir, 'train')
    freshness_non_fresh_val_dir = os.path.join(freshness_non_fresh_fish_dir, 'val')

    classification_dir = os.path.join(datasets_dir, 'Classification')

    for label in classification_labels:
        path =  os.path.join(classification_dir, label)
        classification_fish_paths[label.replace(' ', '_').lower() + '_dir'] = path

    for key, value in classification_fish_paths.items():
        path = os.path.join(value, 'train')
        classification_fish_train_paths[key.replace('_dir', '_train').lower() + '_dir'] = path

    for key, value in classification_fish_paths.items():
        path = os.path.join(value, 'val')
        classification_fish_val_paths[key.replace('_dir', '_val').lower() + '_dir'] = path

    for key, value in classification_fish_train_paths.items():
        classification_fish_train_paths[key] = value.replace('\\', '/')  

    for key, value in classification_fish_val_paths.items():
        classification_fish_val_paths[key] = value.replace('\\', '/')

    if os.path.exists(datasets_dir):
        print("Directory already exist! using existing directory! SET 'clean_dataset=True' to remake the directories.")
        pass
    else:
        os.makedirs(freshness_fresh_train_dir)
        os.makedirs(freshness_fresh_val_dir)
        os.makedirs(freshness_non_fresh_train_dir)
        os.makedirs(freshness_non_fresh_val_dir)

        for key, value in classification_fish_train_paths.items():
            try :
                os.makedirs(value, exist_ok=True)
                print(f"Folder '{key}' created at '{value}'")
            except OSError as e:
                print(f"Error creating folder '{key}' at '{value}': {e}")

        for key, value in classification_fish_val_paths.items():
            try :
                os.makedirs(value, exist_ok=True)
                print(f"Folder '{key}' created at '{value}'")
            except OSError as e:
                print(f"Error creating folder '{key}' at '{value}': {e}")
        print("Train and Val Directory has been created!")
        pass
        
    return classification_fish_train_paths, classification_fish_val_paths, freshness_fresh_train_dir, freshness_fresh_val_dir, freshness_non_fresh_train_dir, freshness_non_fresh_val_dir

def check_files(source):
    folders = []
    files = []

    # Get list of all items (folders, files, etc.) in the directory
    items = os.listdir(source)

    for root, dirnames, filenames in os.walk(source):
        for dirname in dirnames:
            folders.append(os.path.relpath(os.path.join(root, dirname), source))

        for filename in filenames:
            files.append(os.path.relpath(os.path.join(root, filename), source))

def shuffle_data(source_dir, train_dir, val_dir, split_size):
    files=[]
    data_dict = {}
    pathsep		= "\\"
    classification_data = {}

    try:
        for path, dirs, files in os.walk(source_dir):
            for file in files:
                sep_file = os.path.join(path, file)

                if os.path.getsize(sep_file) > 0:
                    key = path.split(pathsep)[len(path.split(pathsep))-1].lower().replace(' ', '_') + "_train_dir".lower()
                # Check if the key already exists in the dictionary
                    if key in classification_data:
                        # If the key exists, append the file path to the existing list
                        classification_data[key].append(sep_file)
                    else:
                        # If the key doesn't exist, initialize it as a list with the file path
                        classification_data[key] = [sep_file]

                else:
                    print(f"File has no weight: {file}. IGNORING!")

        # Now 'classification_data' contains directories as keys and lists of file paths as values
        # print(classification_data)

    except Exception as e:
        print(f"An error occurred: {e}")
        
    for key, value in classification_data.items():
        updated_file_paths = [file.replace('\\', '/') for file in value]
        classification_data[key] = updated_file_paths

    for key in classification_data:
        random.shuffle(classification_data[key])

    # print("Items in classification_fish_train_paths:", classification_fish_train_paths.items())
    # print("Items in classification_fish_val_paths:", classification_fish_train_paths.items())
    # print("Items in classification_data:", classification_data.items())

    # try:
    #     for class_name, files_list in classification_data.items():
    #         dest_dir = classification_fish_train_paths.get(class_name)

    #         if dest_dir is not None:
    #             # Create the destination directory if it doesn't exist
    #             os.makedirs(dest_dir, exist_ok=True)

    #             # Copy files from source directory to the destination directory
    #             for file in files_list:
    #                 filename = os.path.basename(file)
    #                 dest_file = os.path.join(dest_dir, filename)
    #                 copyfile(file, dest_file)
    #                 print(f"Filename: {filename} copied from {file} to {dest_file}")
    #         else:
    #             print(f"No destination path found for class: {class_name}. Skipping copying.")

    # except Exception as e:
    #     print(f"An error occurred during file copying: {e}")
    train_path = {}
    val_path = {}
    try:
        for files_list, train_path, val_path in zip (classification_data.values(), classification_fish_train_paths.values(), classification_fish_val_paths.values()):
            print(files_list)
            print(train_path)
            print(val_path)

            if train_path is not None and val_path is not None:

                # Randomly shuffle the files list
                random.shuffle(files_list)

                # Calculate split point for 80/20 split
                split_point = int(len(files_list) * 0.8)

                # Split files into train and validation sets
                train_files = files_list[:split_point]
                val_files = files_list[split_point:]

                # Copy files to the train folder
                for file in train_files:
                    filename = os.path.basename(file)
                    dest_file = os.path.join(train_path, filename)
                    shutil.copyfile(file, dest_file)
                    print(f"Filename: {filename} copied from {file} to {dest_file} (Train)")

                # Copy files to the validation folder
                for file in val_files:
                    filename = os.path.basename(file)
                    dest_file = os.path.join(val_path, filename)
                    shutil.copyfile(file, dest_file)
                    print(f"Filename: {filename} copied from {file} to {dest_file} (Validation)")

            else:
                print(f"No destination path found for class: . Skipping copying.")

    except Exception as e:
        print(f"An error occurred during file copying: {e}")

 ###########################
    # print(classification_fish_train_paths)

    # keys_list = list(classification_data.keys())
    # random.shuffle(keys_list)

    # train_set_size = int(len(classification_data)*SPLIT_SIZE)

    # train_keys = keys_list[:train_set_size]
    # val_keys = keys_list[train_set_size:]

    # train_data = {key: classification_data[key] for key in train_keys}
    # val_data = {key: classification_data[key] for key in val_keys}

    # # print("Training data keys:", train_data.keys())
    # # print("Validation data keys:", val_data.keys())

    # for (src_key, src_value), (des_key, des_value) in zip(train_data.items(), classification_fish_train_paths.items()):
    #     src_value = train_data[src_key]
    #     des_value = classification_fish_train_paths.get(src_key)
    # if des_value is not None:
    #     for file in src_value:
    #         filename = os.path.basename(file)
    #         des_file = os.path.join(des_key, filename)
    #         copyfile(file, des_file)
    #         print(f"Filename: {filename} successfully copied from {file} to {des_file}")
    # else:
    #     print(f"No destination path found for key: {src_key}. Skipping copying operation.")
    # for subdirectories in os.listdir(source_dir):
    #     subdirectories_path = os.path.join(source_dir, subdirectories)

    #     for filename in os.listdir(subdirectories_path):
    #         file_name = os.path.join(subdirectories_path, filename)

    #         if os.path.getsize(file_name) > 0:
    #             files.append(filename)
    #             data_dict[filename] = file_name
    #         else:
    #             print(f"file has no weight {file_name}. IGNORING!")

    #     train_set_size = int(len(files)*split_size)
    #     shuffled_data = random.sample(files, len(files))

    #     train_data = shuffled_data[0:train_set_size]
    #     val_data = shuffled_data[train_set_size:len(files)]

    #     for file in train_data:
    #         src_file = os.path.join(subdirectories_path, file)
    #         des_file = os.path.join(train_dir, file)
    #         copyfile(src_file, des_file)
    #         print(f"filename : {file} successfully copied from {src_file} -> {des_file}")

    #     for file in val_data:
    #         src_file = os.path.join(source_dir, file)
    #         des_file = os.path.join(val_dir, file)
    #         copyfile(src_file, des_file)
    #         print(f"filename : {file} successfully copied from {src_file} -> {des_file}")
        
    #     pass

def train_val_generator(train_dir, val_dir):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=50,
                                       height_shift_range=50,
                                       shear_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    
    train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                        batch_size=10,
                                                        class_mode='binary',
                                                        target_size=(125, 125))
    
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    val_generator = train_datagen.flow_from_directory(directory=val_dir,
                                                        batch_size=10,
                                                        class_mode='binary',
                                                        target_size=(125, 125))
    return train_generator, val_generator

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(125,125,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

classification_labels = labels_classification_directory_txt()
classification_fish_train_paths, classification_fish_val_paths, freshness_fresh_train_dir, freshness_fresh_val_dir, freshness_non_fresh_train_dir, freshness_non_fresh_val_dir = create_train_val_dir(root_path=root_path, classification_labels=classification_labels)

shuffle_data(classification_data_source_path, classification_fish_train_paths, classification_fish_val_paths, SPLIT_SIZE)

# check_files(classification_data_source_path) #optional

# for rootdir, dirs, files in os.walk(root_path): #check subdirectories?
#     for subdir in dirs:
#         print(os.path.join(root_path, subdir))

# shuffle_data(fresh_path_source, train_fresh_dir, val_fresh_dir, .8) #shuffle data dari data ke dataset training
# shuffle_data(non_fresh_path_source, train_non_fresh_dir, val_non_fresh_dir, .8) #shuffle data dari data ke datasset validasi
 
# train_generator, val_generator = train_val_generator(train_dir, val_dir) #generate data

# print(f"jumlah foto ikan segar = {len(os.listdir(source_path_fresh))}")
# print(f"jumlah foto ikan non segar = {len(os.listdir(source_path_fresh))}")

# model = create_model()
# history = model.fit(train_generator, epochs=100, verbose=1, validation_data=val_generator)
# model.save('Capstone_test_1/saved_model.h5')
# loaded_model = tf.keras.models.load_model('Capstone_test_1/saved_model.h5')

def PREDICT(model, image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(125,125))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0
    
    prediction = model.predict(img_array)
    
    if prediction[0] < 0.5:
        print("ikan termasuk dalam kategori segar")

    else:
        print("ikan termasuk dalam kategori tidak segar")

# image_path1 = 'PREDICT/1.jpg'
# image_path2 = 'PREDICT/2.jpg'
# image_path3 = 'PREDICT/3.jpg'
# PREDICT(loaded_model, image_path1)
# PREDICT(loaded_model, image_path2)
# PREDICT(loaded_model, image_path3)