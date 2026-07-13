import os
import io
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "efficientnet_final.keras")
CLASS_NAMES = ['2021YC', '2022YC', '2023YC', '2024YC', '2025YC']
IMG_SIZE = (224, 224)

AGE_LABELS = {
    '2021YC': '5 year fish',
    '2022YC': '4 year fish',
    '2023YC': '3 year fish',
    '2024YC': '2 year fish',
    '2025YC': '1 year fish'
}

# ── Load Model (once at startup) ────────────────────────
print("Loading model...")
import tensorflow as tf
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

# ── Preprocessing ───────────────────────────────────────
def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img_array = np.expand_dims(img, axis=0)
    return img_array

# ── Routes ──────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        image_bytes = file.read()
        img_array = preprocess_image(image_bytes)
        predictions = model.predict(img_array, verbose=0)[0]
        
        probs = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
        sorted_probs = sorted(probs.items(), key=lambda x: x[-1], reverse=True)
        
        top1_class, top1_prob = sorted_probs[0]
        
        if top1_prob < 0.40:
            return jsonify({
                'success': False,
                'error': 'No fish detected. Please upload a clear Catla fish image.'
            })
        
        top2_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0
        top1_age = AGE_LABELS.get(top1_class, top1_class)
        
        return jsonify({
            'success': True,
            'prediction': {
                'year_class': top1_class,
                'age_label': top1_age,
                'confidence': round(top1_prob * 100, 2),
                'top2_confidence': round((top1_prob + top2_prob) * 100, 2),
                'ranked': [{'class': c, 'age': AGE_LABELS.get(c, c), 'probability': round(p * 100, 2)} for c, p in sorted_probs]
            }
        })
    except Exception as e:
        import traceback
        print("ERROR:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)