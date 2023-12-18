import tflite_support

tflite_codegen --model="TFLite Models/With Metadata/FreshnessModel.tflite" --package_name=org.tensorflow.lite.classify --model_class_name=SeaFest_Freshness --destination=./classify_wrapper