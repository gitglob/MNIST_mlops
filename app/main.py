import pickle
import numpy as np
from fastapi import FastAPI

app = FastAPI()

# Load the trained model from file
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Define a function to make predictions with the model
def predict(data):
    input_data = np.array(data).reshape(1, -1)
    return model.predict(input_data)[0]

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict_endpoint(input_data: list):
    result = predict(input_data)
    return {"prediction": result}
