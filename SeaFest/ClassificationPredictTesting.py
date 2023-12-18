import tensorflow as tf
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np

image_path = "Datasets/Classification/training/Gurame/ikan_ikan_Gurame_0_jpg.rf.4cac66e65aa4404680f23b38ca426bce.jpg"

model = tf.saved_model.load('SeaFest_SavedModels')

def PREDICT(model, image_path):

    infer = model.signatures['serving_default']

    # Image Preprocessing
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Inference
    output = infer(tf.constant(img_array))

    # Access the output layer's name 
    output_layer_name = list(output.keys())[0]

    # Get the predicted class index
    predicted_class_index = np.argmax(output[output_layer_name][0])

    print(output)
    print(predicted_class_index)

    # Define the class labels
    label_class_index = "Balusu (Tenpounder - Ikan Pisang),Bandeng,Belanak (Mullet),Belanak Merah (Ikan Jenggot),Belut Air Tawar,Betok-Puyu (Climbing Perch),Black Spotted Barb -1,Bulan Bulan,Buntal Bintik Hijau,Gabus,Gilt Head Bream,Glass Perchlet,Goby,Grass Carp,Gurame,Gurame Kepala Besar,Hiu Tikus 1,Hiu Tikus 2,Horse Mackerel,Ikan Jenggot,Ikan Lumpur (Mudfish),Ikan Mas,Ikan Mola,Ikan Nyamuk Barat ( Gambusia Affinis ),Indian Carp,Jaguar Guapote,Kakap Merah,Kakap Putih (Sea Bass),Kerapu,Knifefish,Lele,Long-Snouted Pipefish,Longtail Tuna 22,Patin,Perch,Red Sea Bream,Sapu-sapu,Scat Fish (Ikan Argus-Kiper),Senangin,Silver Perch,Sprat Laut Hitam,Tawes,Tenggiri (Spanish Mackerel),Tilapia,Tongkol 3,Tongkol Abu-abu (Longtail Tuna),Trout,Udang"
    label_class_index = label_class_index.split(',')
    
    # Get Predicted Labels
    predicted_class_label = label_class_index[predicted_class_index]

    print("Predicted Class Label:", predicted_class_label)

PREDICT(model, image_path)