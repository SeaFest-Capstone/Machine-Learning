import tensorflow as tf

# Path to TFLite model file
tflite_model_path = "TFLite Models/ClassificationModel.tflite"

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input
print("Input Tensor Details:")
for detail in input_details:
    print(detail)

# Print output
print("\nOutput Tensor Details:")
for detail in output_details:
    print(detail)
