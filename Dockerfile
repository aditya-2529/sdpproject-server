FROM python:3.9-slim  

WORKDIR /app  

COPY . /app  

RUN pip install --no-cache-dir -r requirements.txt  

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

EXPOSE 5000  

CMD ["python", "app.py"]