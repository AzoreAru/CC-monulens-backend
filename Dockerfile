FROM python:3.12-slim

WORKDIR /app

COPY . /app

# Salin key.json ke dalam folder yang sesuai di container
COPY cob/key.json /app/cob/key.json


ENV GOOGLE_APPLICATION_CREDENTIALS="/app/cob/key.json"

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "/app/app.py"]
