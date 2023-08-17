import tensorflow as tf
import numpy as np
import cv2 as cv

class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

MODEL = tf.keras.models.load_model('../model_1')

img = cv.imread(r'D:\Dnk_project\imageye\amitabh_bachchan\Amitabh_Bachchan_2013.jpg')

resized_input_data = tf.image.resize(img, size=(256, 256))
batch_img = np.expand_dims(resized_input_data, 0)
print(batch_img.shape)

prediction = MODEL.predict(batch_img)
disease = class_name[np.argmax(prediction)]
confidence = 100 * np.max(prediction)

print(disease, confidence)
cv.imshow(f'{disease}', img)
cv.waitKey(0)
