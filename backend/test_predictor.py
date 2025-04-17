from app.predictor import SarcasmDetector
import tensorflow as tf

model = SarcasmDetector()
print(model.predict("Oh wow, what a brilliant idea."))
print(model.predict("I am hungry."))
# text = "Oh great, another Monday morning."
# result = detector.predict(text)
# print("Prediction: ", "Sacarstic" if result == 1 else "Not Sarcastic")
