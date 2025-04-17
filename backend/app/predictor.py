import joblib
from .utils import clean_text
from transformers import pipeline
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import torch
import os
from pathlib import Path
# import tensorflow as tf
from overrides import override
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# class SarcasmDetector:
#     def __init__(self):
#         current_file = Path(__file__)
#         model_dir = os.path.join(current_file.parent.parent, "Models/XLNet")
#         self.tokenizer = XLNetTokenizer.from_pretrained(model_dir, local_files_only = True)
#         self.model = XLNetForSequenceClassification.from_pretrained(model_dir)
#         self.model.eval()

#     def predict(self, text: str):
#         inputs = self.tokenizer(text, return_tensors = "pt", truncation = True, padding = True)
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             logits = outputs.logits
#             prediction = torch.argmax(logits, dim = 1).item()
#         return prediction

class BaseTextClassifier:
    def __init__(self, model, tokenizers = None):
        # pre-trained model
        self.model = model

        # dictionary of tokenizers to accomodate to LSTM model that requires multiple tokenizers
        self.tokenizers = tokenizers

    def preprocess(self, text):
        raise NotImplementedError("This method should be implemented by child classes")
    
    def predict(self, text):
        # Preprocess should include the tokenising and padding of input data
        preprocessed_data = self.preprocess(text)
        prediction = self.model.predict(preprocessed_data)
        return prediction
    
class LSTMTextClassifier(BaseTextClassifier):
    def __init__(self, model_path, word_tok_path, char_tok_path, pos_tok_path):
        import pickle

        # Load the LSTM Model
        model = tf.keras.models.load_model(model_path)

        with open(word_tok_path, "rb") as f:
            word_tokenizer = pickle.load(f)
        
        with open(char_tok_path, "rb") as f:
            char_tokenizer = pickle.load(f)
            
        with open(pos_tok_path, "rb") as f:
            pos_tokenizer = pickle.load(f)

        tokeniser_dict = {
            "word": word_tokenizer,
            "char": char_tokenizer,
            "pos": pos_tokenizer
        }

        super.__init__(model, tokeniser_dict)

    @override
    def preprocess(self, text):

        # Tokenization using the different tokenizers
        word_tokens = self.tokenizers['word'].texts_to_sequences([text])
        char_tokens = self.tokenizers['char'].texts_to_sequences([text])
        pos_tokens = self.tokenizers['pos'].texts_to_sequences([text])

        # Padding sequences to ensure uniform input length
        # from tensorflow.keras.preprocessing.sequence import pad_sequences
        word_seq_padded = pad_sequences(word_tokens, padding='post')
        char_seq_padded = pad_sequences(char_tokens, padding='post')
        pos_seq_padded = pad_sequences(pos_tokens, padding='post')

        # Return the preprocessed tokenized data for input into the model
        return [word_seq_padded, char_seq_padded, pos_seq_padded]
    
    
class XLNetTextClassifier(BaseTextClassifier):
    def __init__(self, model, tokenizers=None):
        super().__init__(model, tokenizers)

    def preprocess(self, text, max_length=128):
        tokenizer = self.tokenizers['tokenizer']
        encodings = tokenizer(text, return_tensors = "pt", truncation = True, padding = True).to(self.model.device)

        # Return the tokenized input for the model
        return encodings

    @override
    def predict(self, text):
        inputs = self.preprocess(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim = 1).item()
        return prediction