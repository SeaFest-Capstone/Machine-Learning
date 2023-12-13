import tensorflow as tf
import pathlib

saved_model_dir = 'SeaFest Freshness/'

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
tflite_model = converter.convert()

with open("TFLite Models/Freshness.tflite", "wb") as f:
    f.write(tflite_model)