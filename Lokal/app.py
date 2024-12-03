from flask import Flask, request, jsonify
import os
import tensorflow as tf
from PIL import Image  # Pastikan import Image ini ada untuk menghindari NameError
from io import BytesIO
import numpy as np

app = Flask(__name__)

# Path model yang diunduh atau dipasang di lokal
model_path = "C:\\Primer\\Downloads\\Compressed\\API monulens\\monulens_model\\Xception_V3.h5"  # Sesuaikan dengan path model lokal Anda
monulens_model = tf.saved_model.load(model_path)


# Fungsi preprocessing dan prediksi gambar
def predict_image(image_data):
    try:
        # Load image menggunakan Pillow
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


@app.route("/predict", methods=["POST"])
def predict():
    if "picture" not in request.files:
        return jsonify({"error": "No picture uploaded"}), 400

    picture = request.files["picture"]

    if picture.filename == "":
        return jsonify({"error": "No selected picture"}), 400

    # Cek jika file gambar memiliki ekstensi yang valid
    if not picture.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        return jsonify({"error": "Invalid image file format"}), 400

    if picture:
        # Mendapatkan data gambar
        image_data = picture.read()

        # Mendapatkan prediksi dari gambar yang diupload
        prediction = predict_image(image_data)

        # Mengembalikan hasil prediksi sebagai JSON
        return jsonify(prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
