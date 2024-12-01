from flask import Flask, request, jsonify
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from google.cloud import storage
from PIL import Image
from io import BytesIO
from google.cloud import secretmanager
import uuid

app = Flask(__name__)

# Load the TensorFlow Hub model
from keras.layers import TFSMLayer

my_reloaded_model = TFSMLayer("tf_hub_saved_model", call_endpoint='serving_default')

# Tentukan path folder SavedModel
saved_model_path = "Monulens_Model"

# Memuat model menggunakan tf.saved_model.load()
monulens_model = tf.saved_model.load(saved_model_path)

# Set the path to your service account key (key.json)
SERVICE_ACCOUNT_KEY_PATH = 'key.json'

# Initialize Google Cloud Storage client with the service account key
storage_client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_KEY_PATH)

# Set your Google Cloud Storage bucket name
BUCKET_NAME = os.environ.get('CLOUD_STORAGE_BUCKET', 'monulens-bucket')



def predict_image(BUCKET_NAME, blob_name):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    img_data = blob.download_as_bytes()

    # Load dan preprocess gambar
    img = Image.open(BytesIO(img_data)).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Gunakan TFSMLayer untuk prediksi
    prediction = my_reloaded_model(img_array)

    # Dapatkan 3 prediksi teratas
    top3_classes_idx = np.argsort(prediction.numpy()[0])[-3:][::-1]
    class_names = [
        "Patung Pahlawan",
        "Monumen Nasional",
        "Monumen Pembebasan Irian Barat",
        "Monumen Selamat Datang",
        "Patung Pangeran Diponegoro",
        "Monumen IKADA",
        "Monumen Perjuangan Senen",
        "Patung R.A. Kartini",
        "Patung Kuda Arjuna Wijaya",
        "Patung M.H. Thamrin",
        "Patung Persahabatan",
    ]

    predicted_class_names = [class_names[idx] for idx in top3_classes_idx]
    predicted_probabilities = [float(prediction.numpy()[0][idx]) for idx in top3_classes_idx]

    return {
        "predicted_class_names": predicted_class_names,
        "predicted_probabilities": predicted_probabilities
    }


def upload_to_bucket(file_storage, blob_name):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)

    # Save the file storage content to a temporary file
    temp_file_path = f"/tmp/{uuid.uuid4()}.jpg"  # Use a unique file name
    file_storage.save(temp_file_path)

    # Upload the temporary file to Google Cloud Storage
    blob.upload_from_filename(temp_file_path)

    # Clean up the temporary file
    os.remove(temp_file_path)
    

def get_service_key():
    client = secretmanager.SecretManagerServiceClient()
    secret_name = "projects/YOUR_PROJECT_ID/secrets/monulens-service-key/versions/latest"
    response = client.access_secret_version(request={"name": secret_name})
    secret_payload = response.payload.data.decode("UTF-8")
    return secret_payload

# Simpan key.json ke file sementara
service_key = get_service_key()
with open("key.json", "w") as f:
    f.write(service_key)

@app.route('/predict', methods=['POST'])
def predict():
    if 'picture' not in request.files:
        return jsonify({"error": "No picture uploaded"}), 400

    picture = request.files['picture']

    if picture.filename == '':
        return jsonify({"error": "No selected picture"}), 400

    if picture:
        # Generate a unique blob name using uuid
        blob_name = f"uploads/{uuid.uuid4()}.jpg"

        # Upload the image to Google Cloud Storage
        upload_to_bucket(picture, blob_name)

        # Get prediction using the processing function
        prediction = predict_image(blob_name)

        # Return the prediction result as JSON
        return jsonify(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
