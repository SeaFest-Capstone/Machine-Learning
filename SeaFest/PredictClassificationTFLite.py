import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="SeaFest_SavedModels/5. InceptionV3_SeaFestClassification.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape'] 
input_data = 'Raw Data/Classification/Longtail Tuna/Screenshot_2020-08-21 longtail tuna - Bing images(154).png'

img = tf.keras.preprocessing.image.load_img(input_data, target_size=(299, 299))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array/255.0

# Set input tensor to the loaded model
interpreter.set_tensor(input_details[0]['index'], img_array)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
softmax_output = tf.nn.softmax(output_data)

print("Output probabilities:", softmax_output)

predicted_class_index = np.argmax(softmax_output)

print(predicted_class_index)

# Define the class labels
label_class_index = "Bandeng, Belanak, Gabus, Gurame, Kakap Merah, Lele, Longtail Tuna, Patin, Tenggiri"
label_class_index = label_class_index.split(', ')

print(label_class_index)
# Get Predicted Labels
predicted_class_label = label_class_index[predicted_class_index]

print("Predicted Class Label:", predicted_class_label)
