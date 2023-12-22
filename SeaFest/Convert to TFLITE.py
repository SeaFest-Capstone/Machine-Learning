import tensorflow as tf

# Saved Model Path
saved_model_dir = tf.keras.models.load_model('SeaFest_SavedModels/3. Fix2_ClassificationModel/Fix2SeaFestClassification')

# Converting Models
converter = tf.lite.TFLiteConverter.from_keras_model(saved_model_dir)
# converter.target_spec.supported_ops = [
#   tf.lite.OpsSet.TFLITE_BUILTINS,
#   tf.lite.OpsSet.SELECT_TF_OPS
# ]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

open("SeaFest_SavedModels/3. Fix2_ClassificationModel/Compressed_SeaFest_Classification V1.0.3.tflite", "wb").write(tflite_model)