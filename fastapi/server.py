import uvicorn
from fastapi import FastAPI, UploadFile, File
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()
MODEL = tf.keras.models.load_model('../model_1')
class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def img_to_numpy(bytes) -> np.ndarray:
    image = np.array(Image.open(BytesIO(bytes)))
    resized_input_data = tf.image.resize(image, size=(256, 256))
    return resized_input_data

@app.post("/")
async def predict(
        file: UploadFile = File(...)
):
    img = img_to_numpy(await file.read())
    batch_img = np.expand_dims(img, 0)
    prediction = MODEL.predict(batch_img)

    disease = class_name[np.argmax(prediction)]
    confidence = 100 * np.max(prediction)

    return {
        'class': disease,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
