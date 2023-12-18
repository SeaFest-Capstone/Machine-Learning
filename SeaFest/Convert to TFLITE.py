import tensorflow as tf
import pathlib

saved_model_dir = 'SeaFest_SavedModels/Freshness_2'

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("TFLite Models/FreshnessModel2.tflite", "wb") as f:
    f.write(tflite_model)