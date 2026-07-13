import os
import cv2
import math
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "efficientnet_final.tflite")
CLASS_NAMES = ['2021YC', '2022YC', '2023YC', '2024YC', '2025YC']
IMG_SIZE = (224, 224)

AGE_LABELS = {
    '2021YC': '5 year fish',
    '2022YC': '4 year fish',
    '2023YC': '3 year fish',
    '2024YC': '2 year fish',
    '2025YC': '1 year fish'
}

# ── Thresholds ──
CONFIDENCE_THRESHOLD = 0.50      # Top class must be at least 50% confident
GAP_THRESHOLD = 0.20             # Gap between top-1 and top-2 must be 20%
MAX_ENTROPY = 1.0                # Lower = more certain prediction

print("Loading TFLite model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded!")

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = preprocess_input(img.astype(np.float32))
    return np.expand_dims(img, axis=0)

def predict_tflite(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

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
        predictions = predict_tflite(img_array)
        
        probs = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        top1_class, top1_prob = sorted_probs[0]
        top2_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0
        
        # ── FISH DETECTION CHECKS ──
        
        # Check 1: Confidence threshold
        if top1_prob < CONFIDENCE_THRESHOLD:
            return jsonify({
                'success': False,
                'error': 'No fish detected. Please upload a clear Catla fish image.'
            })
        
        # Check 2: Gap between top-1 and top-2
        if (top1_prob - top2_prob) < GAP_THRESHOLD:
            return jsonify({
                'success': False,
                'error': 'Uncertain prediction. Please upload a clearer Catla fish image.'
            })
        
        # Check 3: Entropy (flat distribution = not a fish)
        entropy = -sum(p * math.log(p + 1e-10) for p in predictions)
        if entropy > MAX_ENTROPY:
            return jsonify({
                'success': False,
                'error': 'Image does not appear to be a fish. Please upload a Catla fish image.'
            })
        
        # ── PASSED ALL CHECKS ──
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