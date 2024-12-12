FROM python:3.12-slim

WORKDIR /app

# Menyalin seluruh file aplikasi ke dalam image
COPY . /app

# Menyalin serviceAccount.json ke dalam container
COPY cob/serviceAccount.json /app/cob/serviceAccount.json

# Set environment variable untuk menggunakan serviceAccount.json
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/cob/serviceAccount.json"

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Menjalankan aplikasi Flask menggunakan gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
