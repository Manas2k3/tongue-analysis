import tensorflow as tf
import numpy as np
import os
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)

# === CONFIG ===
MODEL_PATH = "tongue_disease_classifier_v1.h5"
TEST_DIR = "dataset-split/test"  # Define the path to your test directory
IMG_SIZE = (224, 224)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load model and class labels ===
print("🔄 Loading model and class labels...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# 💡 Dynamically get class names from the directory structure
try:
    # We sort the list to ensure the order is consistent (alphabetical)
    # This matches the behavior of Keras's flow_from_directory
    class_names = sorted(os.listdir(TEST_DIR))
    print(f"✅ Model loaded. Classes found: {class_names}")
except FileNotFoundError:
    print(f"❌ ERROR: The directory '{TEST_DIR}' was not found. Please check the path.")
    # Exit or set a default list if you want the app to still run
    class_names = []


@app.route('/predict', methods=['POST'])
def predict():
    if not class_names:
        return jsonify({"error": "Server is not configured correctly; class names not loaded."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Preprocess the image
        img = load_img(filepath, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_tensor)
        pred_index = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_index]) * 100

        result = {
            "predicted_class": class_names[pred_index],
            "confidence": round(confidence, 2)
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"message": "Tongue Analysis API is running 🚀"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)