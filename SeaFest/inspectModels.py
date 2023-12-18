import tensorflow as tf

# Path to SavedModel
model_path = 'SeaFest_SavedModels/SeaFest_Classification_SavedModels'

# Load Model
loaded_model = tf.saved_model.load(model_path)

# Print model signatures
print("Model Signatures:")
print(loaded_model.signatures)

# Access model components (layers, variables)
print("\nModel Layers:")
for layer in loaded_model.signatures['serving_default'].structured_input_signature[1].values():
    print(layer.name, layer.shape)

# Accessing input and output tensors
# Replace 'input_tensor_name' and 'output_tensor_name' with tensor name from your model
input_tensor_name = 'input_1'
output_tensor_name = 'dense_4'

input_tensor = loaded_model.signatures['serving_default'].structured_input_signature[1][input_tensor_name]
output_tensor = loaded_model.signatures['serving_default'].structured_outputs[output_tensor_name]

print("\nInput Tensor Shape:", input_tensor.shape)
print("Output Tensor Shape:", output_tensor.shape)