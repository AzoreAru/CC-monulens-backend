import os
import firebase_admin
from firebase_admin import credentials, initialize_app, storage
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify

# Inisialisasi Flask app
app = Flask(__name__)

# Path to your Firebase service account key
SERVICE_ACCOUNT_KEY_PATH = "/app/cob/serviceAccount.json"

# Initialize Firebase Admin SDK
cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
firebase_app = initialize_app(cred, {
    'storageBucket': 'monulens-backend.firebasestorage.app'
})

# Firebase Storage Bucket reference
bucket = storage.bucket()

def download_model_from_firebase(model_path):
    """
    Downloads a TensorFlow SavedModel from Firebase Storage and loads it.

    Args:
        model_path (str): Path to the model folder in Firebase Storage.

    Returns:
        Loaded TensorFlow SavedModel.
    """
    # Local path to store the downloaded model
    local_model_path = "/home/c247b4ky2964/cob"
    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path)

    # List blobs (files) in the specified Firebase Storage path
    blobs = bucket.list_blobs(prefix=model_path)
    for blob in blobs:
        # Skip directories
        if blob.name.endswith('/'):
            continue

        # Determine local file path
        relative_path = blob.name.replace(model_path, "").lstrip("/")
        local_file_path = os.path.join(local_model_path, relative_path)

        # Ensure local directories exist
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the blob to the local file
        blob.download_to_filename(local_file_path)
        print(f"Downloaded {blob.name} to {local_file_path}")

    # Load and return the TensorFlow SavedModel
    model = tf.saved_model.load(local_model_path)
    return model

# Download and load the model
model_path = "model/Xception_V3"  # Path to the model folder in Firebase Storage
monulens_model = download_model_from_firebase(model_path)

# Fungsi untuk memeriksa struktur model dan nama layer output
def print_model_structure(model):
    print("Model structure:")
    for layer in model.signatures['serving_default'].structured_outputs:
        print(layer)

# Panggil fungsi untuk menampilkan struktur model
print_model_structure(monulens_model)


# Fungsi untuk memproses gambar
def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))  # Pastikan ukuran sesuai dengan input model Anda
    image = np.array(image) / 255.0  # Normalisasi piksel gambar
    image = np.expand_dims(image, axis=0)  # Menambahkan batch dimension
    return image

def run_inference(image):
    infer = monulens_model.signatures['serving_default']
    
    # Mengonversi image menjadi float32 sebelum inferensi
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # Melakukan inferensi
    output = infer(image)
    
    # Debugging: Tampilkan output dan shape
    print(f"Model output: {output}")
    
    # Output model bisa berupa dictionary, dan kita perlu memilih tensor yang sesuai
    output_tensor = output['output_0']  # Ganti 'output_0' dengan nama layer output yang benar jika diperlukan
    
    # Debugging: Tampilkan bentuk output
    print(f"Output shape: {output_tensor.shape}")
    
    # Jika output berupa tensor dengan lebih dari satu dimensi
    output_array = output_tensor.numpy()
    print(f"Output array: {output_array}")
    
    # Jika output berupa array 2D, gunakan np.argmax untuk memilih kelas
    if len(output_array.shape) == 2:  # Output berupa (batch_size, num_classes)
        predicted_class = np.argmax(output_array, axis=1)[0]
        predicted_prob = np.max(output_array, axis=1)[0]
    else:  # Output 1D atau kelas tunggal
        predicted_class = np.argmax(output_array)
        predicted_prob = np.max(output_array)

    # Menghitung persentase
    predicted_prob_percent = predicted_prob * 100
    
    # Debugging: Tampilkan prediksi dan probabilitas
    print(f"Predicted class: {predicted_class}, Probability: {predicted_prob_percent:.2f}%")
    
    return predicted_class, predicted_prob_percent

# Fungsi untuk mengambil deskripsi dan sejarah monumen dari Firestore
def get_monument_info(prediction):
    """
    Mengambil informasi monumen dari Firestore berdasarkan ID prediksi.

    Args:
        prediction (int): ID prediksi dari model.

    Returns:
        dict: Informasi monumen dari Firestore.
    """
    from google.cloud import firestore
    db = firestore.Client()

    # Peta (mapping) prediksi ke ID dokumen Firestore
    prediction_to_doc_id = {
        0: "nNpgNZPYpnRV0AmRWF4K",  # ID dokumen untuk prediksi 0
        1: "Bhap9hXDJogZM6NFbbue",  # ID dokumen untuk prediksi 1
        2: "rfI86TEJpJdQxjWyyfFt",  # ID dokumen untuk prediksi 2
        3: "P7AuYOPvAUZGhfgEg8e3",
        4: "Dq6sIU0uEyYhLoQquNkj",
        5: "IpzuNNBJAvIpMJqqtQgU",
        6: "WlwiqWkdKy6losMrJZVY",
        7: "ow2ONc8nPVhQkoAIVzh6",
        8: "cHowg7dut2vCfJ1BFaJY",
        9: "cvp3EI1DHO77epM6MZnr",
        10: "A0jOUeRT8jLCeN7tbr8s",
        11: "W6UYArYlwVC4rQkZRtrc"
    }

    # Ambil ID dokumen berdasarkan prediksi
    doc_id = prediction_to_doc_id.get(prediction)

    # Jika prediksi tidak memiliki mapping
    if not doc_id:
        return {"error": "Prediction not mapped to any monument in Firestore"}

    # Ambil dokumen dari Firestore
    monument_ref = db.collection('monuments').document(doc_id)
    doc = monument_ref.get()

    # Jika dokumen ditemukan, kembalikan data
    if doc.exists:
        return doc.to_dict()

    # Jika dokumen tidak ditemukan
    return {"error": "Monument not found"}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Membaca gambar dari file yang di-upload
    image_bytes = file.read()
    image = preprocess_image(image_bytes)

    # Melakukan inferensi pada gambar
    try:
        predicted_class, predicted_prob_percent = run_inference(image)
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

    # Prediksi (misalnya model memberikan indeks atau nama monumen)
    predicted_monument = str(predicted_class)  # Anda bisa mengganti ini dengan ID atau nama monumen yang sesuai

    # Ambil deskripsi dan sejarah dari Firestore
    monument_info = get_monument_info(predicted_class)

    # Return hasil prediksi dan informasi monumen
    return jsonify({
        "prediction": predicted_monument,
        "probability": f"{predicted_prob_percent:.2f}%",
        "monument_info": monument_info
    })

# API untuk mengambil detail monumen berdasarkan ID
@app.route('/monuments/<string:monument_id>', methods=['GET'])
def get_monument_by_id(monument_id):
    """
    Mengambil data monumen berdasarkan ID dokumen di Firestore.

    Args:
        monument_id (str): ID dokumen di Firestore.

    Returns:
        JSON: Data monumen atau pesan error jika tidak ditemukan.
    """
    from google.cloud import firestore
    db = firestore.Client()

    # Ambil dokumen dari koleksi "monuments"
    monument_ref = db.collection('monuments').document(monument_id)
    doc = monument_ref.get()

    # Jika dokumen ditemukan, kembalikan data
    if doc.exists:
        return jsonify(doc.to_dict())
    
    # Jika dokumen tidak ditemukan
    return jsonify({"error": "Monument not found"}), 404


# API untuk menampilkan daftar semua monumen
@app.route('/monuments', methods=['GET'])
def list_monuments():
    """
    Mengambil daftar semua monumen dari Firestore.

    Returns:
        JSON: Daftar semua monumen.
    """
    from google.cloud import firestore
    db = firestore.Client()

    # Ambil semua dokumen dari koleksi "monuments"
    monuments = db.collection('monuments').stream()

    # Ubah setiap dokumen menjadi dictionary
    monument_list = []
    for monument in monuments:
        data = monument.to_dict()
        data['id'] = monument.id  # Sertakan ID dokumen dalam respons
        monument_list.append(data)

    # Kembalikan daftar monumen
    return jsonify(monument_list)


if __name__ == '__main__':
    app.run(debug=True)
