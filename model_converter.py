import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the old model
model = load_model("tongue_disease_classifier_v1.h5", compile=False)

# Save in the new Keras-native format instead (AVOID .h5)
model.save("clean_model.h5")
