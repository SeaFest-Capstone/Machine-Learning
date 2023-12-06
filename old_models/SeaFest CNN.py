import os
import zipfile
import random
import shutil
import tensorflow as tf
from shutil import copyfile
# import matplotlib.pyplot as plt
import numpy as np

source_path = 'fresh-nofresh/'
root_path = 'fresh_nofresh_dataset/'

def create_train_val_dir(root_path):
    if os.path.exists(root_path):
        shutil.rmtree(root_path)

    source_path_fresh = os.path.join(source_path, 'fresh')
    source_path_non_fresh = os.path.join(source_path, 'non_fresh')

    fresh_path_source = os.path.join(source_path, 'fresh')
    non_fresh_path_source = os.path.join(source_path, 'non_fresh')

    train_dir = os.path.join(root_path, 'train')
    val_dir = os.path.join(root_path, 'val')

    train_fresh_dir = os.path.join(train_dir, 'fresh')
    train_non_fresh_dir = os.path.join(train_dir, 'non_fresh')

    val_fresh_dir = os.path.join(val_dir, 'fresh') 
    val_non_fresh_dir = os.path.join(val_dir, 'non_fresh') 
    os.makedirs(train_fresh_dir)
    os.makedirs(train_non_fresh_dir)
    os.makedirs(val_fresh_dir)
    os.makedirs(val_non_fresh_dir)

    pass

def check_files():
    if not os.path.exists(source_path_fresh):
        print(f"The path '{source_path_fresh}' does not exist.")
    else:
        print(f"Number of fresh fish photos = {len(os.listdir(source_path_fresh))}")

    if not os.path.exists(source_path_non_fresh):
        print(f"The path '{source_path_non_fresh}' does not exist.")
    else:
        print(f"Number of non-fresh fish photos = {len(os.listdir(source_path_non_fresh))}")

def create_train_val_dir(root_path):
    if os.path.exists(root_path):
        shutil.rmtree(root_path)

    source_path_fresh = os.path.join(source_path, 'fresh')
    source_path_non_fresh = os.path.join(source_path, 'non_fresh')

    fresh_path_source = os.path.join(source_path, 'fresh')
    non_fresh_path_source = os.path.join(source_path, 'non_fresh')

    train_dir = os.path.join(root_path, 'train')
    val_dir = os.path.join(root_path, 'val')

    train_fresh_dir = os.path.join(train_dir, 'fresh')
    train_non_fresh_dir = os.path.join(train_dir, 'non_fresh')

    val_fresh_dir = os.path.join(val_dir, 'fresh') 
    val_non_fresh_dir = os.path.join(val_dir, 'non_fresh') 
    os.makedirs(train_fresh_dir)
    os.makedirs(train_non_fresh_dir)
    os.makedirs(val_fresh_dir)
    os.makedirs(val_non_fresh_dir)

    pass

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

source_path_fresh = os.path.join(source_path, 'fresh')
source_path_non_fresh = os.path.join(source_path, 'non_fresh')

fresh_path_source = os.path.join(source_path, 'fresh')
non_fresh_path_source = os.path.join(source_path, 'non_fresh')

train_dir = os.path.join(root_path, 'train')
val_dir = os.path.join(root_path, 'val')

train_fresh_dir = os.path.join(train_dir, 'fresh')
train_non_fresh_dir = os.path.join(train_dir, 'non_fresh')

val_fresh_dir = os.path.join(val_dir, 'fresh') 
val_non_fresh_dir = os.path.join(val_dir, 'non_fresh') 

# create_train_val_dir(root_path) #first

# check_files() #optional

# for rootdir, dirs, files in os.walk(root_path): #check subdirectories?
#     for subdir in dirs:
#         print(os.path.join(root_path, subdir))

# shuffle_data(fresh_path_source, train_fresh_dir, val_fresh_dir, .8) #shuffle data dari data ke dataset training
# shuffle_data(non_fresh_path_source, train_non_fresh_dir, val_non_fresh_dir, .8) #shuffle data dari data ke datasset validasi
 
train_generator, val_generator = train_val_generator(train_dir, val_dir) #generate data

# print(f"jumlah foto ikan segar = {len(os.listdir(source_path_fresh))}")
# print(f"jumlah foto ikan non segar = {len(os.listdir(source_path_fresh))}")

model = create_model()

history = model.fit(train_generator, epochs=100, verbose=1, validation_data=val_generator)

# model.save('Capstone_test_1/saved_model.h5')

# loaded_model = tf.keras.models.load_model('Capstone_test_1/saved_model.h5')

# def PREDICT(model, image_path):
#     img = tf.keras.preprocessing.image.load_img(image_path, target_size=(125,125))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)

#     img_array = img_array / 255.0
    
#     prediction = model.predict(img_array)
    
#     if prediction[0] < 0.5:
#         print("ikan termasuk dalam kategori segar")

#     else:
#         print("ikan termasuk dalam kategori tidak segar")

# image_path1 = 'PREDICT/1.jpg'
# image_path2 = 'PREDICT/2.jpg'
# image_path3 = 'PREDICT/3.jpg'
# PREDICT(loaded_model, image_path1)
# PREDICT(loaded_model, image_path2)
# PREDICT(loaded_model, image_path3)

# def PREDICT(model):
#     image_input = []
#     image_path = 'Capstone_test_1/PREDICT/'
#     for filename in image_path:
#         filenames = os.path.join(image_path, filename)
#         image_input.append(filename)
    
#     for i in image_input:
#         img = tf.keras.preprocessing.image.load_img(image_input[int(i)], target_size=(125,125))
#         img_array = tf.keras.preprocessing.image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)

#         img_array = img_array / 255.0
        
#         prediction = model.predict(img_array)
#         if prediction[i] < 0.5:
#             print("fish is fresh")

#         else:
#             print("not fresh")