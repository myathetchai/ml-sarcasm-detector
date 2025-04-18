from flask import Flask, request, jsonify
from .predictor import SarcasmDetector 
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict_sarcasm():
    data = request.json
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400
     
    text = data["message"]
    detector = SarcasmDetector()  
    result = detector.predict(text)  
    return jsonify({"sarcastic": result})
