# app.py


from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import tensorflow as tf  # or use your preferred deep learning library

app = Flask(__name__)

# Load your pre-trained deep learning model
# Replace 'model_path' with the path to your model file
model = tf.keras.models.load_model(r"D:\Dnk_project\Deep_learning\tf_model\model_1")


def preprocess_image(image):
    # Resize, preprocess, and convert the image to an array
    # Modify this function according to your model's input requirements
    resized_input_data = tf.image.resize(image, size=(256, 256))
    batch_img = np.expand_dims(resized_input_data, 0)
    return batch_img


@app.route("/predict", methods=["POST"])
def predict_potato_disease():
    if "image" not in request.files:
        return jsonify({"error": "No image in the request"}), 400

    image = request.files["image"]
    img = Image.open(image)

    processed_img = preprocess_image(img)

    # Make a prediction using your deep learning model
    prediction = model.predict(processed_img)
    # Example: prediction will be a list of probabilities for each class.

    # You can convert the prediction to a human-readable response
    # based on your model's classes and return it to the client.
    # Example:
    class_name = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
    result = {
        "prediction": class_name[np.argmax(prediction)],
        "probabilities": 100 * np.max(prediction)
    }

    return jsonify(result)


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


if __name__ == "__main__":
    app.run(host='192.168.1.14', port=5000, debug=True)
