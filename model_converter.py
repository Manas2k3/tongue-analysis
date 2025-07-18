from tensorflow.keras.models import load_model
import tensorflow as tf

model = tf.keras.models.load_model("tongue_disease_classifier_v1.h5", compile=False)
model.save("tongue_disease_classifier_v2.keras")  # without save_format