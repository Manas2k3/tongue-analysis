from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from pyngrok import ngrok  # 🔗 For public URL

app = Flask(__name__)

# === CONFIG ===
MODEL_PATH = "tongue_disease_classifier_v1.h5"
TEST_DIR = "tongue-split/test"
IMG_SIZE = (224, 224)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load model and class labels ===
print("🔄 Loading model and class labels...")
model = load_model(MODEL_PATH)

datagen = ImageDataGenerator(rescale=1./255)
dummy_gen = datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)
class_names = list(dummy_gen.class_indices.keys())
print(f"✅ Model loaded. Classes: {class_names}")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        img = load_img(filepath, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0), dtype=tf.float32)

        predictions = model.predict(img_tensor)
        pred_index = np.argmax(predictions)
        confidence = float(predictions[0][pred_index]) * 100

        result = {
            "predicted_class": class_names[pred_index],
            "confidence": round(confidence, 2)
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"message": "Tongue Analysis API is running 🚀"})

if __name__ == '__main__':
    # 🔓 Start ngrok tunnel
    port = 5000
    public_url = ngrok.connect(port)
    print(f"✨ Ngrok URL: {public_url}")
    
    # 🏃 Run Flask
    app.run(port=port)
