import os
import random
import shutil
import tensorflow as tf
import shutil
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(10000) # Limit Recursive Printing to Avoid RecursionError

root_path = ''
source_path = os.path.join(root_path, 'Raw Data')
freshness_data_source_path = os.path.join(source_path, 'Freshness')
clean_dataset = True
SPLIT_SIZE = 0.8

# Create Training and Validation Directory with its Associated Labels, and Save Training and Validation Directory Path
def create_train_val_dir(root_path):
    freshness_training_fish_paths = {}
    freshness_validation_fish_paths = {}

    datasets_dir = os.path.join(root_path, 'Datasets')
    freshness_dir = os.path.join(datasets_dir, 'Freshness')

    if clean_dataset==True and os.path.exists(freshness_dir)==True :
        shutil.rmtree(freshness_dir)

    freshness_training_dir = os.path.join(freshness_dir, 'training')
    freshness_validation_dir = os.path.join(freshness_dir, 'validation')

    freshness_fresh_training_dir = os.path.join(freshness_training_dir, 'fresh')
    freshness_fresh_validation_dir = os.path.join(freshness_validation_dir, 'fresh')

    freshness_nonfresh_training_dir = os.path.join(freshness_training_dir, 'non fresh')
    freshness_nonfresh_validation_dir = os.path.join(freshness_validation_dir, 'non fresh')

    freshness_training_fish_paths['training_fresh_dir'] = freshness_fresh_training_dir
    freshness_training_fish_paths['training_nonfresh_dir'] = freshness_nonfresh_training_dir
    freshness_validation_fish_paths['validation_fresh_dir'] = freshness_fresh_validation_dir
    freshness_validation_fish_paths['validation_nonfresh_dir'] = freshness_nonfresh_validation_dir

    if clean_dataset == False:
        print("Directory already exist! using existing directory! SET 'clean_dataset=True' to remake the directories.")
        pass
    else:
        os.makedirs(freshness_fresh_training_dir)
        os.makedirs(freshness_fresh_validation_dir)
        os.makedirs(freshness_nonfresh_training_dir)
        os.makedirs(freshness_nonfresh_validation_dir)

    return freshness_training_dir, freshness_validation_dir, freshness_training_fish_paths, freshness_validation_fish_paths

# Copy Split and Shuffle data from Source Directory to Datasets Directory
def copy_split_shuffle_data(freshness_data_source_path, freshness_training_fish_paths, freshness_validation_fish_paths, split_size):
    files=[]
    pathsep		= "\\"
    freshness_data = {}

    try:
        for path, dirs, files in os.walk(freshness_data_source_path):
            for file in files:
                sep_file = os.path.join(path, file)

                if os.path.getsize(sep_file) > 0:
                    key = path.split(pathsep)[len(path.split(pathsep))-1].lower().replace(' ', '_') + "_train_dir".lower()
                    if key in freshness_data:
                        freshness_data[key].append(sep_file)
                    else:
                        freshness_data[key] = [sep_file]
                else:
                    print(f"File has no weight: {file}. IGNORING!")
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        for files_list, train_path, val_path in zip (freshness_data.values(), freshness_training_fish_paths.values(), freshness_validation_fish_paths.values()):
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

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                    rotation_range=20,
                                    width_shift_range=20,
                                    height_shift_range=30,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    brightness_range=(0.5, 1.5),
                                    vertical_flip=True,
                                    horizontal_flip=True,
                                    fill_mode='nearest')


    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                    rotation_range=20,
                                    zoom_range=0.2,
                                    vertical_flip=True,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

    train_generators = train_datagen.flow_from_directory(
                            directory=train_dir,
                            target_size=(250,250),
                            batch_size=64,
                            class_mode='binary',
                        )

    val_generators = val_datagen.flow_from_directory(
                            directory=val_dir,
                            batch_size=32,
                            class_mode='binary',
                            target_size=(250, 250)
                        )

    return train_generators, val_generators

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
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(250,250,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Create Datasets Directory, Scan and Save Paths for copy_split_shuffle_data and train_val_generator
freshness_training_dir, freshness_validation_dir, freshness_training_fish_paths, freshness_validation_fish_paths = create_train_val_dir(root_path=root_path)

# Copy Split and Shuffle Data from Source to Datasets
if clean_dataset== True:
    copy_split_shuffle_data(freshness_data_source_path, freshness_training_fish_paths, freshness_validation_fish_paths, SPLIT_SIZE)

# ImageDataGenerator to Prevent Overfitting
train_generators, val_generators = train_val_generator(freshness_training_dir, freshness_validation_dir) #generate data

# Initialize Callbacks
lrs_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    'SeaFest_SavedModels/FixSeaFestFreshness_BestModel.h5',
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min'
                )

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="SeaFest_SavedModels/FixSeaFestFreshnessLogs",
                                                        histogram_freq=1,
                                                        write_graph=True,
                                                        update_freq='epoch'
                                                        )

# Build the Model
model = create_model()

# Reset Weight for Multiple Run
initial_weights = model.get_weights()
model.set_weights(initial_weights)

# Train the Model
history = model.fit(train_generators, epochs=100, verbose=1, validation_data=val_generators, callbacks=[lrs_callback, tensorboard_callback, model_checkpoint])

# Saving the Model
model.save('SeaFest_SavedModels/FixSeaFestFreshness')
model.save('SeaFest_SavedModels/FixSeaFestFreshness.h5')

# Plotting Model Training Performance
plot_training(history)

