import tensorflow as tf

model = tf.keras.models.load_model('Capstone_test_1/saved_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("SeaFest_CNN.tflite", "wb").write(tflite_model)