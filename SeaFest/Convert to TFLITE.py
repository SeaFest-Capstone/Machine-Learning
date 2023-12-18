import tensorflow as tf
import pathlib

# Saved Model Path
saved_model_dir = 'SeaFest_SavedModels/Freshness_2'

# Converting Models
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Saving Models to TFLite
with open("TFLite Models/FreshnessModel2.tflite", "wb") as f:
    f.write(tflite_model)