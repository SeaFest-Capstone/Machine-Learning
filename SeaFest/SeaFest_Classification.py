import os
import random
import shutil
import tensorflow as tf
import shutil
import matplotlib.pyplot as plt
import sys
import requests
sys.setrecursionlimit(10000) # Limit Recursive Printing to Avoid RecursionError

root_path = ''
source_path = os.path.join(root_path, 'Raw Data')
classification_data_source_path = os.path.join(source_path, 'Classification')
clean_dataset = True
renew_model = True
model_name = "inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
SPLIT_SIZE = 0.7

# Download InceptionResNetV2 Models
def get_models(model_name):
    # URL to InceptionResNetV2 Model
    url = "https://storage.googleapis.com/tensorflow/keras-applications/inception_resnet_v2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
    output_path = model_name

    # HTTP Request to Download InceptionResNetV2 Model
    response = requests.get(url)

    # Save InceptionResNetV2 Model
    with open(output_path, 'wb') as f:
        f.write(response.content)

# Get Data Labels
def labels_classification_to_txt():
    outputfile	= "classification_labels.txt" # TXT File Output
    folder		= "Raw Data/Classification"	# Directory for Getting Labels
    pathsep		= "\\"
    classification_labels = [] # Empty List to Contains Labels
    
    # Scan Directory
    try:
        for path,dirs,files in os.walk(folder):
            sep = path.split(pathsep)[len(path.split(pathsep))-1]
            # print(sep)
            classification_labels.append(sep)

    except IndexError:
        pass

    classification_labels.pop(0) # Remove First Appended Label (subdir name)

    # Save Labels to TXT
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

# Create Training and Validation Directory with its Associated Labels, and Save Training and Validation Directory Path
def create_train_val_dir(root_path, classification_labels):
    classification_training_fish_paths = {}
    classification_validation_fish_paths = {}

    datasets_dir = os.path.join(root_path, 'Datasets')
    classification_dir = os.path.join(datasets_dir, 'Classification')

    if clean_dataset==True and os.path.exists(classification_dir)==True :
        shutil.rmtree(classification_dir)

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

    if clean_dataset == False and os.path.exists(classification_dir):
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

# Copy Split and Shuffle data from Source Directory to Datasets Directory
def copy_split_shuffle_data(source_dir,  classification_training_fish_paths, classification_validation_fish_paths, split_size):
    files=[]
    pathsep	= "\\"
    classification_data = {}
    train_path = {}
    val_path = {}

    try:
        for path, dirs, files in os.walk(source_dir):
            for file in files:
                sep_file = os.path.join(path, file)

                if os.path.getsize(sep_file) > 0:
                    key = path.split(pathsep)[len(path.split(pathsep))-1].lower().replace(' ', '_') + "_train_dir".lower()
                    
                    if key in classification_data:
                        classification_data[key].append(sep_file)
                    else:
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

    try:
        for files_list, train_path, val_path in zip (classification_data.values(),  classification_training_fish_paths.values(), classification_validation_fish_paths.values()):
            print(files_list)
            print(train_path)
            print(val_path)

            if train_path is not None and val_path is not None:
                random.shuffle(files_list)

                split_point = int(len(files_list) * split_size)
                train_files = files_list[:split_point]
                val_files = files_list[split_point:]

                for file in train_files:
                    filename = os.path.basename(file)
                    dest_file = os.path.join(train_path, filename)
                    shutil.copyfile(file, dest_file)
                    print(f"Filename: {filename} copied from {file} to {dest_file} (Train)")

                for file in val_files:
                    filename = os.path.basename(file)
                    dest_file = os.path.join(val_path, filename)
                    shutil.copyfile(file, dest_file)
                    print(f"Filename: {filename} copied from {file} to {dest_file} (Validation)")

            else:
                print(f"No destination path found for class: . Skipping copying.")

    except Exception as e:
        print(f"An error occurred during file copying: {e}")

# ImageDataGenerator to Augment Images (Prevent Overfitting)
def train_val_generator(train_dir, val_dir):
 
    species_train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                    rotation_range=20,
                                    width_shift_range=20,
                                    height_shift_range=30,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    brightness_range=(0.5, 1.5),
                                    vertical_flip=True,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
    
    species_val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                    rotation_range=20,
                                    zoom_range=0.2,
                                    vertical_flip=True,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
    
    species_train_generators = species_train_datagen.flow_from_directory(
                                directory=train_dir,
                                target_size=(250,250),
                                batch_size=64,
                                class_mode='categorical'
                            )

    species_val_generators= species_val_datagen.flow_from_directory(
                                directory=val_dir,
                                batch_size=32,
                                class_mode='categorical',
                                target_size=(250, 250)
                            )

    return species_train_generators, species_val_generators

# LearningRateScheduler Callbacks
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

# Plotting Training History
def plot_training(history):
    # Plot training and validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training and validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Build the Model
def build_model(model_name):
    local_weights_file = model_name
    InceptionResNetV2_Models = tf.keras.applications.InceptionResNetV2(input_shape=(250, 250, 3), 
                                                                        include_top=False, 
                                                                        weights=None)
    
    InceptionResNetV2_Models.load_weights(local_weights_file)
    # InceptionResNetV2_Models.summary()

    for layer in InceptionResNetV2_Models.layers:
        layer.trainable = False

    last_layer = InceptionResNetV2_Models.get_layer('block17_15_mixed')
    last_output = last_layer.output

    x = tf.keras.layers.Flatten()(last_output)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(9, activation='softmax')(x)

    model = tf.keras.Model(InceptionResNetV2_Models.input, x) 
    model.summary()
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Download InceptionResNetV2
if renew_model == True:
    get_models(model_name)
else:
    pass

# Get Data Labels
classification_labels = labels_classification_to_txt()

# Create Datasets Directory, Scan and Save Paths for copy_split_shuffle_data and train_val_generator
classification_training_dir, classification_validation_dir, classification_training_fish_paths, classification_validation_fish_paths = create_train_val_dir(root_path=root_path, classification_labels=classification_labels)

# Copy Split and Shuffle Data from Source to Datasets
if clean_dataset == True:
    copy_split_shuffle_data(classification_data_source_path, classification_training_fish_paths, classification_validation_fish_paths, SPLIT_SIZE)

# ImageDataGenerator to Prevent Overfitting
species_train_generators, species_val_generators = train_val_generator(classification_training_dir, classification_validation_dir)

# Initialize Learning Rate Scheduler
lrs_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    'SeaFest_SavedModels/FixSeaFestClassification_BestModel.h5',
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min'
                )

# Build the Model
model = build_model(model_name)

# Train the Model
history = model.fit(species_train_generators, epochs=100, verbose=1, validation_data=species_val_generators, callbacks=[lrs_callback, model_checkpoint])

# Saving the Model
model.save('SeaFest_SavedModels/FixSeaFestClassification')
model.save('SeaFest_SavedModels/FixSeaFestClassification.h5')

# Plotting Model Training Performance
plot_training(history)
