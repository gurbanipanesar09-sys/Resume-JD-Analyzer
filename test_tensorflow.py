import tensorflow as tf

print("TensorFlow imported successfully")
print("TensorFlow version:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())

tensor = tf.constant([10, 20, 30])
print("Test tensor:", tensor)