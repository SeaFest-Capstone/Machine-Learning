import tensorflow as tf
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np

image_path = "Datasets/Classification/training/Lele/00255ef6-05cd-46d4-8724-694847af29dc-860mm.jpg"

model = tf.keras.models.load_model('SeaFest_SavedModels/2. InceptionV3_ClassificationModel/InceptionV3_SeaFestClassification_BestModel.h5')

def PREDICT(model, image_path):

    # infer = model.signatures['serving_default']

    # Image Preprocessing
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Inference
    # output = infer(tf.constant(img_array))
    output = model.predict(img_array)

    # Access the output layer's name 
    # output_layer_name = list(output.keys())[0]

    # Get the predicted class index
    # predicted_class_index = np.argmax(output[output_layer_name][0])
    predicted_class_index = np.argmax(output)

    print(output)
    print(predicted_class_index)

    # Define the class labels
    label_class_index = "Bandeng, Belanak, Gabus, Gurame, Kakap Merah, Lele, Longtail Tuna, Patin, Tenggiri"
    label_class_index = label_class_index.split(', ')
    
    # Get Predicted Labels
    predicted_class_label = label_class_index[predicted_class_index]

    print("Predicted Class Label:", predicted_class_label)

PREDICT(model, image_path)