import joblib
from .utils import clean_text
from transformers import XLNetTokenizer, XLNetForSequenceClassification, BertTokenizer, BertForSequenceClassification
import torch
import os
from pathlib import Path
import requests
import gdown
import logging
import xgboost as xgb
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.current_file = Path(__file__)
        self.models_dir = os.path.join(self.current_file.parent.parent, "Models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.model_configs = {
             "xlnet": {
                "type": "transformers",
                "files": {
                    "config.json": "1oYmvswHUW27Csr5jc9ooO1VlfHcpW87t",
                    "model.safetensors": "1ibgyYJbnaBcN8VEmW0iSsHcYbaYpY5iS",
                    "tokenizer_config.json": "1-mCphBFPFOnv4Sb2trcKqZQ4COtoToOi",
                    "special_tokens_map.json": "1VdgaH9mYH-A_sxpe_3j0RH3JqipBfJNy",
                    "spiece.model": "19l7uxapF-kRAHgw-71xRCCZRXnXRQBsj"
                }
            }
         }
        '''
        # Define model configurations
        self.model_configs = {
            
            "logistic_regression": {
                "type": "sklearn",
                "files": {
                    "model.pkl": "1KadCDS4pYn6Rw10iLDbzORJ0edviH9Ok"  # bert_logreg_model.pkl
                }
            },
        
            "xgboost": {
                "type": "xgboost",
                "files": {
                    "model.pkl": "1xgl6bBJ92rJ8veNVvZDNkiV1ngX_Ohvp"  # xgb_model.pkl
                }
            },
            "xlnet": {
                "type": "transformers",
                "files": {
                    "config.json": "1oYmvswHUW27Csr5jc9ooO1VlfHcpW87t",
                    "model.safetensors": "1ibgyYJbnaBcN8VEmW0iSsHcYbaYpY5iS",
                    "tokenizer_config.json": "1-mCphBFPFOnv4Sb2trcKqZQ4COtoToOi",
                    "special_tokens_map.json": "1VdgaH9mYH-A_sxpe_3j0RH3JqipBfJNy"
                }
            },
            
            "bigru": {
                "type": "keras_multi_input",  # Changed type to indicate multiple inputs
                "files": {
                    "model.h5": "1Oot5rGOZMQescLUErbpB-SbopaxJDFEv",
                    "char_tokenizer.pkl": "1tSuZsIqSeZJvPM5RbNX3rURQnb2ckusu",
                    "word_tokenizer.pkl": "1YoOAJy0XPNrozmxn94MMCs0pu9TGeU_y",
                    "pos_tokenizer.pkl": "1-sHIqWXdEVpfd6HmrvvgwW1_kQMcPBlc",
                    "tfidf_vectorizer.pkl": "1pSwqWgbzQqEyS2L8DaO1OqM6_86armCq"
                }
            
            }
        }
        '''
            
            
          
        
        
        # Initialize models dictionary
        self.models = {}
        self.tokenizers = {}
        
        # Download and load all models
        self._initialize_models()

    def _download_models(self):
        """Download all model files from Google Drive if they don't exist locally"""
        for model_name, config in self.model_configs.items():
            model_dir = os.path.join(self.models_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            for filename, file_id in config["files"].items():
                file_path = os.path.join(model_dir, filename)
                if not os.path.exists(file_path):
                    try:
                        logger.info(f"Downloading {model_name} - {filename}...")
                        url = f"https://drive.google.com/uc?id={file_id}"
                        output = file_path
                        gdown.download(url, output, quiet=False)
                        logger.info(f"Successfully downloaded {model_name} - {filename}")
                    except Exception as e:
                        logger.error(f"Error downloading {model_name} - {filename}: {str(e)}")
                        raise

    def _initialize_models(self):
        """Initialize all models after downloading"""
        self._download_models()
        
        for model_name, config in self.model_configs.items():
            model_dir = os.path.join(self.models_dir, model_name)
            
            try:
                if config["type"] == "sklearn":
                    self.models[model_name] = joblib.load(os.path.join(model_dir, "model.pkl"))
                elif config["type"] == "xgboost":
                    self.models[model_name] = xgb.Booster()
                    self.models[model_name].load_model(os.path.join(model_dir, "model.json"))
                elif config["type"] == "transformers":
                    if model_name == "xlnet":
                        self.tokenizers[model_name] = XLNetTokenizer.from_pretrained(model_dir, local_files_only=True)
                        self.models[model_name] = XLNetForSequenceClassification.from_pretrained(model_dir)
                    else:  # bert
                        self.tokenizers[model_name] = BertTokenizer.from_pretrained(model_dir, local_files_only=True)
                        self.models[model_name] = BertForSequenceClassification.from_pretrained(model_dir)
                    self.models[model_name].eval()
                elif config["type"] == "keras":
                    self.models[model_name] = load_model(os.path.join(model_dir, "model.h5"))
                    self.tokenizers[model_name] = joblib.load(os.path.join(model_dir, "tokenizer.pkl"))
                elif config["type"] == "keras_multi_input":
                    # Process multiple inputs for BiGRU
                    self.tokenizers[f"{model_name}_char"] = joblib.load(os.path.join(model_dir, "char_tokenizer.pkl"))
                    self.tokenizers[f"{model_name}_word"] = joblib.load(os.path.join(model_dir, "word_tokenizer.pkl"))
                    self.tokenizers[f"{model_name}_pos"] = joblib.load(os.path.join(model_dir, "pos_tokenizer.pkl"))
                    self.tokenizers[f"{model_name}_tfidf"] = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
                    self.models[model_name] = load_model(os.path.join(model_dir, "model.h5"))
                
                logger.info(f"Successfully loaded {model_name}")
            except Exception as e:
                logger.error(f"Error loading {model_name}: {str(e)}")
                raise

    def predict(self, text: str, model_name: str = None):
        """Make prediction using specified model or all models"""
        predictions = {}
        
        if model_name:
            models_to_use = {model_name: self.model_configs[model_name]}
        else:
            models_to_use = self.model_configs
            
        for name, config in models_to_use.items():
            try:
                if config["type"] == "sklearn":
                    # Preprocess text and make prediction
                    processed_text = clean_text(text)
                    prediction = self.models[name].predict([processed_text])[0]
                    predictions[name] = prediction
                elif config["type"] == "xgboost":
                    # Preprocess text and make prediction
                    processed_text = clean_text(text)
                    dmatrix = xgb.DMatrix(np.array([processed_text]))
                    prediction = self.models[name].predict(dmatrix)[0]
                    predictions[name] = 1 if prediction > 0.5 else 0
                elif config["type"] == "transformers":
                    # Tokenize and make prediction
                    inputs = self.tokenizers[name](text, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        outputs = self.models[name](**inputs)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=1).numpy()[0]
                        prediction = torch.argmax(logits, dim=1).item()
                    predictions[name] = prediction
                    print(f"\n>>> Input: {text}")
                    print(f"XLNet Probabilities: Not Sarcastic: {probs[0]:.4f}, Sarcastic: {probs[1]:.4f}")
                elif config["type"] == "keras":
                    # Tokenize and make prediction
                    sequence = self.tokenizers[name].texts_to_sequences([text])
                    padded_sequence = pad_sequences(sequence, maxlen=100)  # Adjust maxlen as needed
                    prediction = self.models[name].predict(padded_sequence)[0][0]
                    predictions[name] = 1 if prediction > 0.5 else 0
                elif config["type"] == "keras_multi_input":
                    # Process multiple inputs for BiGRU
                    inputs = []
                    # Character-level input
                    char_sequences = self.tokenizers[f"{name}_char"].texts_to_sequences([text])
                    char_padded = pad_sequences(char_sequences, maxlen=100)
                    inputs.append(char_padded)
                    
                    # Word-level input
                    word_sequences = self.tokenizers[f"{name}_word"].texts_to_sequences([text])
                    word_padded = pad_sequences(word_sequences, maxlen=50)
                    inputs.append(word_padded)
                    
                    # POS features
                    pos_features = self.tokenizers[f"{name}_pos"].transform([text])
                    inputs.append(pos_features)
                    
                    # TF-IDF features
                    tfidf_features = self.tokenizers[f"{name}_tfidf"].transform([text])
                    inputs.append(tfidf_features.toarray())
                    
                    # Make prediction with all inputs
                    prediction = self.models[name].predict(inputs)[0][0]
                    predictions[name] = 1 if prediction > 0.5 else 0
                    
            except Exception as e:
                logger.error(f"Error making prediction with {name}: {str(e)}")
                predictions[name] = None

        
        return predictions

class SarcasmDetector:
    def __init__(self):
        self.model_manager = ModelManager()

    def predict(self, text: str, model_name: str = None):
        """Make prediction using specified model or all models"""
        return self.model_manager.predict(text, model_name)
    