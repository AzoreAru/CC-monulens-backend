from flask import Flask, request, jsonify
import os
import tensorflow as tf
from google.cloud import storage
from PIL import Image
from io import BytesIO
import uuid
import numpy as np

app = Flask(__name__)

# Set the path to your service account key (key.json)
SERVICE_ACCOUNT_KEY_PATH = "/app/cob/key.json"

# Initialize Google Cloud Storage client with the service account key
storage_client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_KEY_PATH)

# Set your Google Cloud Storage bucket name
BUCKET_NAME = os.environ.get("CLOUD_STORAGE_BUCKET", "monulens-bucket4")

def download_model_from_gcs(model_path, bucket_name):
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=model_path)

    local_model_path = "/home/c247b4ky2964/cob"  # Temporary folder for saved model
    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path)

    for blob in blobs:
        # Download each file in the model folder to the local path
        local_file_path = os.path.join(local_model_path, blob.name.replace(model_path, "").lstrip("/"))
        if not os.path.exists(os.path.dirname(local_file_path)):
            os.makedirs(os.path.dirname(local_file_path))
        blob.download_to_filename(local_file_path)

    # Now, load the SavedModel
    model = tf.saved_model.load(local_model_path)
    return model

# Download and load the model
model_path = "model/Xception_V3"  # Path to the SavedModel in GCS
monulens_model = download_model_from_gcs(model_path, BUCKET_NAME)

# Function to preprocess and predict the image using the model
def predict_image(image_data):
    try:
        # Pastikan image_data adalah dalam bentuk byte stream
        img = Image.open(BytesIO(image_data)).convert("RGB").resize((224, 224))
    except Exception as e:
        return {"error": f"Error loading image: {e}"}, 500

    img_array = np.array(img) / 255.0  # Normalisasi gambar
    img_array = np.expand_dims(img_array, axis=0)  # Menambahkan dimensi batch

    # Gunakan model untuk prediksi
    infer = monulens_model.signatures["serving_default"]
    prediction = infer(tf.convert_to_tensor(img_array, dtype=tf.float32))

    # Debugging: Tampilkan key yang tersedia pada prediction
    print("Available keys in prediction:", prediction.keys())

    # Dinamis: Ambil key pertama yang tersedia untuk hasil prediksi
    key_name = list(prediction.keys())[0]  # Ambil key pertama dari dictionary
    print(f"Using key: {key_name}")

    # Ekstraksi hasil prediksi berdasarkan key
    pred_scores = prediction[key_name]
    top3_classes_idx = np.argsort(pred_scores[0])[-3:][::-1]

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
    predicted_probabilities = [float(pred_scores[0][idx]) for idx in top3_classes_idx]

    return {
        "predicted_class_names": predicted_class_names,
        "predicted_probabilities": predicted_probabilities,
    }

# Function to upload image to GCS
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

@app.route("/predict", methods=["POST"])
def predict():
    if "picture" not in request.files:
        return jsonify({"error": "No picture uploaded"}), 400

    picture = request.files["picture"]

    if picture.filename == "":
        return jsonify({"error": "No selected picture"}), 400

    # Cek apakah file yang diunggah adalah gambar
    if not picture.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({"error": "Invalid image file format"}), 400

    try:
        # Membaca gambar sebagai byte
        image_data = picture.read()
        
        # Pastikan image_data adalah tipe byte
        if isinstance(image_data, bytes):
            prediction = predict_image(image_data)  # Panggil fungsi predict_image dengan data byte
        else:
            raise ValueError("Data yang diterima bukan dalam bentuk byte")

        # Kembalikan hasil prediksi dalam format JSON
        return jsonify(prediction)

    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
