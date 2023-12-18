import tensorflow as tf
import numpy as np

# Path to Images
image_path = "Datasets/Freshness/validation/fresh/20200517_084015.jpg"
image_path1 = "Datasets/Freshness/validation/non fresh/DSC00509.JPG"

# Load Model
model = tf.keras.models.load_model('SeaFest_SavedModels/Freshness_2')

def PREDICT(model, image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(250,250))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0
    
    prediction = model.predict(img_array)
    
    if prediction[0] < 0.5:
        print("ikan termasuk dalam kategori TIDAK segar")

    else:
        print("ikan termasuk dalam kategori segar")

PREDICT(model, image_path)
PREDICT(model, image_path1)