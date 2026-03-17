from fastapi import FastAPI
import pickle

app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "House Price Prediction API"}

@app.post("/predict")
def predict(area: float, bedrooms: int):
    prediction = model.predict([[area, bedrooms]])
    return {"predicted_price": prediction[0]}