# Gunakan base image Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy file ke container
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY tf_hub_saved_model/ tf_hub_saved_model/
COPY Monulens_Model.h5 Monulens_Model.h5

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port Flask
EXPOSE 8080

# Jalankan aplikasi
CMD ["python", "app.py"]
