from flask import Flask, request, jsonify
from .model import predict_sacarsm

app = Flask(__name__)

def predict():
    data = request.json
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400
    
    text = data["message"]
    result = predict_sacarsm(text)
    return jsonify({"sacarstic": result})
