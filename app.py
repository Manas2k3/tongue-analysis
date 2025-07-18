import tensorflow as tf
import numpy as np
import os
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)

# === CONFIG ===
MODEL_PATH = "tongue_disease_classifier_v1.h5"
IMG_SIZE = (224, 224)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load model and class labels ===
print("🔄 Loading model and class labels...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# 💡 FIX: Hardcode the class names directly
#    Replace these with your actual class names in the correct order.
class_names = ['atrophic_glossitis', 'black_hairy_tongue', 'geographic_tongue', 'strawberry_tongue']

print(f"✅ Model loaded. Classes: {class_names}")


@app.route('/predict', methods=['POST'])
def predict():
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
        img_tensor = np.expand_dims(img_array, axis=0) # No need for tf.convert_to_tensor here

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