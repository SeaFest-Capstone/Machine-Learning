import tensorflow as tf

# Replace these values with your metadata information
metadata = {
    "description": "Seafest Classification Model | Users : Hx3 M15",
    "version": "1.0",
    "author": "SeaFest Testing Model | Users : Hx3 M15",
    # Add more metadata as needed
}

# Your HDF5 file path
hdf5_file_path = 'Capstone_test_1/saved_model.h5'
tflite_model_path = 'converted_model.tflite'

# Convert the HDF5 model to TensorFlow Lite model
# Load the model
model = tf.keras.models.load_model(hdf5_file_path)

# Convert to TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Write the TFLite model to a file
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

# Attach metadata as an associated file
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Replace with your metadata file path
metadata_file_path = 'metadata.txt'

with open(metadata_file_path, 'w') as meta_file:
    for key, value in metadata.items():
        meta_file.write(f"{key}: {value}\n")

# Add the metadata file as an associated file
interpreter.add_tensor_with_metadata_file(
    tensor_index=interpreter.get_input_details()[0]['index'],
    filename=metadata_file_path
)

# Save the modified TFLite model
modified_tflite_model_path = 'converted_model_with_metadata.tflite'
with open(modified_tflite_model_path, 'wb') as f:
    f.write(interpreter.tensor(interpreter.get_input_details()[0]['index'])().tobytes())

print("Metadata added as an associated file to the TensorFlow Lite model.")
