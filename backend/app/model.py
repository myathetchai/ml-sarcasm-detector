import joblib
from .utils import clean_text

model = joblib.load("app/model.pkl") 
    
def predict_sacarsm(text):
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    return bool(prediction)