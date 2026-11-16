import mlflow.sklearn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Modèle téléchargé depuis DagsHub (au build de l'image Docker)
model = mlflow.sklearn.load_model("model")

app = FastAPI(title="Digits classifier")


class DigitImage(BaseModel):
    # Image 8x8 aplatie : 64 valeurs de pixels entre 0 et 16
    pixels: list[float] = Field(min_length=64, max_length=64)


@app.post("/predict")
def predict(image: DigitImage):
    prediction = model.predict(np.array([image.pixels]))
    return {"prediction": int(prediction[0])}