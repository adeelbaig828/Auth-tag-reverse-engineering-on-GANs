import io
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model(".//Model//model_V1_.h5")

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image from the request
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))

    # Preprocess the image
    img_width, img_height = 224, 224
    image = image.resize((img_width, img_height))
    image = image.convert('RGB')  # Convert to RGB format
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict the class of the image
    prediction = model.predict(image_array)[0][0]

    # Return the predicted class as a response
    if prediction < 0.5:
        result = {'class': 'Artificial'}
    else:
        result = {'class': 'Human'}

    return jsonify(result)

if __name__ == '__main__':
    app.run()
