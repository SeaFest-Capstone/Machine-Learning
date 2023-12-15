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
            # print(sep)
            classification_labels.append(sep)

    except IndexError:
        pass

    classification_labels.pop(0)

    #SAVE TO TXT
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
    classification_training_fish_paths = {}
    classification_validation_fish_paths = {}

    datasets_dir = os.path.join(root_path, 'Datasets')
    
    if clean_dataset==True and os.path.exists(datasets_dir)==True :
        shutil.rmtree(datasets_dir)

    classification_dir = os.path.join(datasets_dir, 'Classification')

    classification_training_dir = os.path.join(classification_dir, 'training')
    classification_validation_dir = os.path.join(classification_dir, 'validation')

    for label in classification_labels:
        path =  os.path.join(classification_training_dir, label)
        classification_training_fish_paths[label.replace(' ', '_').lower() + '_dir'] = path
    # for key, value in classification_training_fish_dir.items():
    #     print(f"{key}, {value}")

    for label in classification_labels:
        path =  os.path.join(classification_validation_dir, label)
        classification_validation_fish_paths[label.replace(' ', '_').lower() + '_dir'] = path
    # for key, value in classification_validation_fish_paths.items():
    #     print(f"{key}, {value}")

    for key, value in classification_training_fish_paths.items():
        classification_training_fish_paths[key] = value.replace('\\', '/')
    # for key, value in classification_training_fish_dir.items():
    #     print(f"{key}, {value}")    

    for key, value in classification_validation_fish_paths.items():
        classification_validation_fish_paths[key] = value.replace('\\', '/')
    # for key, value in classification_validation_fish_paths.items():
    #     print(f"{key}, {value}")
    print(classification_training_fish_paths)
    print(classification_validation_fish_paths)

    if os.path.exists(datasets_dir):
        print("Directory already exist! using existing directory! SET 'clean_dataset=True' to remake the directories.")
        pass
    else:

        for key, value in classification_training_fish_paths.items():
            try :
                os.makedirs(value, exist_ok=True)
                print(f"Folder '{key}' created at '{value}'")
            except OSError as e:
                print(f"Error creating folder '{key}' at '{value}': {e}")

        for key, value in classification_validation_fish_paths.items():
            try :
                os.makedirs(value, exist_ok=True)
                print(f"Folder '{key}' created at '{value}'")
            except OSError as e:
                print(f"Error creating folder '{key}' at '{value}': {e}")
        print("Train and Val Directory has been created!")
        pass
    
    return classification_training_dir, classification_validation_dir, classification_training_fish_paths, classification_validation_fish_paths

def check_files(source):
    folders = []
    files = []

    items = os.listdir(source)

    for root, dirnames, filenames in os.walk(source):
        for dirname in dirnames:
            folders.append(os.path.relpath(os.path.join(root, dirname), source))

        for filename in filenames:
            files.append(os.path.relpath(os.path.join(root, filename), source))

def copy_split_shuffle_data(source_dir,  classification_training_fish_paths, classification_validation_fish_paths, split_size):
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

    except Exception as e:
        print(f"An error occurred: {e}")
        
    for key, value in classification_data.items():
        updated_file_paths = [file.replace('\\', '/') for file in value]
        classification_data[key] = updated_file_paths

    for key in classification_data:
        random.shuffle(classification_data[key])

    train_path = {}
    val_path = {}
    try:
        for files_list, train_path, val_path in zip (classification_data.values(),  classification_training_fish_paths.values(), classification_validation_fish_paths.values()):
            print(files_list)
            print(train_path)
            print(val_path)

            if train_path is not None and val_path is not None:

                # Randomly shuffle the files list
                random.shuffle(files_list)

                # Calculate split point for SPLIT_SIZE
                split_point = int(len(files_list) * split_size)

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

def train_val_generator(train_dir, val_dir):
 
    species_train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=50,
                                    height_shift_range=50,
                                    shear_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
    
    species_val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    species_train_generators = species_train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(150,150),
        batch_size=64,
        class_mode='categorical',
    )

    species_val_generators= species_val_datagen.flow_from_directory(
        directory=val_dir,
        batch_size=20,
        class_mode='categorical',
        target_size=(150, 150)
    )

    return species_train_generators, species_val_generators

def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

def plot_training(history):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(9, activation='softmax')
    ])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

classification_labels = labels_classification_directory_txt()

classification_training_dir, classification_validation_dir, classification_training_fish_paths, classification_validation_fish_paths = create_train_val_dir(root_path=root_path, classification_labels=classification_labels)

if clean_dataset== True:
    copy_split_shuffle_data(classification_data_source_path, classification_training_fish_paths, classification_validation_fish_paths, SPLIT_SIZE)

species_train_generators, species_val_generators = train_val_generator(classification_training_dir, classification_validation_dir) #generate data

lrs_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

model = create_model()
initial_weights = model.get_weights()
model.set_weights(initial_weights)
history = model.fit(species_train_generators, epochs=100, verbose=1, validation_data=species_val_generators)
plot_training(history)
model.save('SeaFest')
sys.exit()


# check_files(classification_data_source_path) #optional
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