import os
import zipfile
import random
import shutil
import tensorflow as tf
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np

root_path = ''
source_path = os.path.join(root_path, 'Raw Data')
classification_data_source_path = os.path.join(source_path, 'Classification')
freshness_data_source_path = os.path.join(source_path, 'Freshness')
clean_dataset = False

def labels_classification_directory_txt():
    # outputfile	= "classification_labels.txt"
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
    #     for path,dirs,files in os.walk(folder):
    #         sep = path.split(pathsep)[len(path.split(pathsep))-1]
    #         print(sep)
    #         classification_labels.append(sep)
    #     pass

    classification_labels.pop(0)
    return classification_labels

    # SAVE TO TXT
    # try: 
    #     with open(outputfile, "x") as txtfile:		
    #         for path,dirs,files in os.walk(folder):
    #             sep = path.split(pathsep)[len(path.split(pathsep))-1]
    #             print(sep)
    #             txtfile.write("%s\n" % sep)
    #     txtfile.close()

    # except FileExistsError:
    #     os.remove(outputfile)
    #     with open(outputfile, "x") as txtfile:	
    #         for path,dirs,files in os.walk(folder):
    #             sep = path.split(pathsep)[len(path.split(pathsep))-1]
    #             print(sep)
    #             txtfile.write("%s\n" % sep)
    #     txtfile.close()
    #     pass   

def create_train_val_dir(root_path, classification_labels):
    classification_fish_paths = {}
    classification_fish_train_paths = {}
    classification_fish_val_paths = {}

    datasets_dir = os.path.join(root_path, 'Datasets')
    
    if clean_dataset==True:
        shutil.rmtree(datasets_dir)

    freshness_fish_dir = os.path.join(datasets_dir, 'Freshness')

    freshness_fresh_fish_dir = os.path.join(freshness_fish_dir, 'Fresh')
    freshness_fresh_train_dir = os.path.join(freshness_fresh_fish_dir, 'Train')
    freshness_fresh_val_dir = os.path.join(freshness_fresh_fish_dir, 'Val') 

    freshness_non_fresh_fish_dir = os.path.join(freshness_fish_dir, 'Non Fresh')
    freshness_non_fresh_train_dir = os.path.join(freshness_non_fresh_fish_dir, 'Train')
    freshness_non_fresh_val_dir = os.path.join(freshness_non_fresh_fish_dir, 'Val')

    classification_dir = os.path.join(datasets_dir, 'Classification')

    for label in classification_labels:
        path =  os.path.join(classification_dir, label)
        classification_fish_paths[label.replace(' ', '_').lower() + '_dir'] = path
    for key, value in classification_fish_paths.items():
        print(f"{key}, {value}")

    for key, value in classification_fish_paths.items():
        path = os.path.join(value, 'Train')
        classification_fish_train_paths[key.replace('_dir', '_train').lower() + '_dir'] = path
    for key, value in classification_fish_train_paths.items():
        print(f"{key}, {value}")

    for key, value in classification_fish_paths.items():
        path = os.path.join(value, 'val')
        classification_fish_val_paths[key.replace('_dir', '_val').lower() + '_dir'] = path
    for key, value in classification_fish_val_paths.items():
        print(f"{key}, {value}")

    for key, value in classification_fish_train_paths.items():
        classification_fish_train_paths[key] = value.replace('\\', '/')
    for key, value in classification_fish_train_paths.items():
        print(f"{key}, {value}")    

    for key, value in classification_fish_val_paths.items():
        classification_fish_val_paths[key] = value.replace('\\', '/')
    for key, value in classification_fish_val_paths.items():
        print(f"{key}, {value}")

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

    for filename in os.listdir(source_dir):
        file_name = os.path.join(source_dir, filename)
        if os.path.getsize(file_name) > 0:
            files.append(filename)
        else:
            print(f"file has no weight {file_name}. IGNORING!")

        train_set_size = int(len(files)*split_size)
        shuffled_data = random.sample(files, len(files))

        train_data = shuffled_data[0:train_set_size]
        val_data = shuffled_data[train_set_size:len(files)]

        for file in train_data:
            src_file = os.path.join(source_dir, file)
            des_file = os.path.join(train_dir, file)
            copyfile(src_file, des_file)
            print(f"filename : {file} successfully copied from {src_file} -> {des_file}")

        for file in val_data:
            src_file = os.path.join(source_dir, file)
            des_file = os.path.join(val_dir, file)
            copyfile(src_file, des_file)
            print(f"filename : {file} successfully copied from {src_file} -> {des_file}")
        
        pass

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
