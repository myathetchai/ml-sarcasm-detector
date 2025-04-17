import joblib
from .utils import clean_text
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import torch
import os
from pathlib import Path

class SarcasmDetector:
    def __init__(self):
        current_file = Path(__file__)
        model_dir = os.path.join(current_file.parent.parent, "Models/XLNet")
        # model_dir = "C:/Users/andre/OneDrive - National University of Singapore/Desktop/Software Projects/Bias Tele Bot/ml-sarcasm-detector/backend/Models/XLNet"
        self.tokenizer = XLNetTokenizer.from_pretrained(model_dir, local_files_only = True)
        self.model = XLNetForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors = "pt", truncation = True, padding = True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim = 1).item()
        return prediction
    