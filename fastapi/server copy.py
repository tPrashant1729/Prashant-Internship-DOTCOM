import uvicorn
from fastapi import FastAPI, UploadFile, File
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

endpoint = "http://localhost:8501/v1/model/model_potatos:predict"
class_name = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]


def img_to_numpy(bytes) -> np.ndarray:
    image = np.array(Image.open(BytesIO(bytes)))
    resized_input_data = tf.image.resize(image, size=(256, 256))
    return resized_input_data


@app.post("/")
async def predict(file: UploadFile = File(...)):
    img = img_to_numpy(await file.read())
    batch_img = np.expand_dims(img, 0)

    json_data = {"instances": batch_img.tolist()}
    response = requests.post(endpoint, json = json_data)

    pass
    # disease = class_name[np.argmax(prediction)]
    # confidence = 100 * np.max(prediction)

    # return {
    #     'class': disease,
    #     'confidence': float(confidence)
    # }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
