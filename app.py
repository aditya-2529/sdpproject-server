from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
import jwt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

import os
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

model = load_model('facial_expression_model.h5')
face_cascade = cv2.CascadeClassifier('cascade.xml')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_and_prepare_face(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None

    x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, -1)
    face = np.expand_dims(face, 0)
    return face

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MongoDB configuration
app.config["MONGO_URI"] = os.getenv("MONGODB_URI", "mongodb+srv://ranjanaditya2901:SDPPROJECT%40@cluster0.frhovdb.mongodb.net/emotion-api")
mongo = PyMongo(app)

# JWT configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")


os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json()
    
    if not data or not data.get("email") or not data.get("password"):
        return jsonify({"error": "Missing required fields"}), 400
    
    if mongo.db.users.find_one({"email": data["email"]}):
        return jsonify({"error": "Email already exists"}), 400
    
    hashed_password = generate_password_hash(data["password"])
    user = {
        "name": data.get("name", ""),
        "email": data["email"],
        "password": hashed_password,
        "img":"",
        "created_at": datetime.utcnow()
    }
    
    mongo.db.users.insert_one(user)
    
    return jsonify({"message": "User registered successfully"}), 201

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    
    if not data or not data.get("email") or not data.get("password"):
        return jsonify({"error": "Missing required fields"}), 400
    
    user = mongo.db.users.find_one({"email": data["email"]})
    
    if not user or not check_password_hash(user["password"], data["password"]):
        return jsonify({"error": "Invalid credentials"}), 401
    
    token = jwt.encode({
        "user_id": str(user["_id"]),
        "email": user["email"],
        "exp": datetime.utcnow() + timedelta(days=1)
    }, JWT_SECRET)
    
    return jsonify({
        "token": token,
        "user": {
            "id": str(user["_id"]),
            "name": user.get("name", ""),
            "email": user["email"]
        }
    })

@app.route("/api/auth/verify", methods=["GET"])
def verify_token():
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "No token provided"}), 401
    
    token = auth_header.split(" ")[1]
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user = mongo.db.users.find_one({"_id": ObjectId(payload["user_id"])})
        
        if not user:
            raise Exception("User not found")
        
        return jsonify({
            "user": {
                "id": str(user["_id"]),
                "name": user.get("name", ""),
                "email": user["email"]
            }
        })
    except Exception as e:
        return jsonify({"error": "Invalid token"}), 401

@app.route('/api/auth/predict', methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    d = request.form['imageName']
    id = request.form['id']
    if file.filename == '':
        return jsonify({'error': 'Empty file name'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        mongo.db.users.update_one({"_id":ObjectId(id)},{"$push": {"img":d}})

        face_input = detect_and_prepare_face(filepath)
        if face_input is None:
            os.remove(filepath)
            return jsonify({'error': 'No face detected'}), 200  # Send 200 so JS still receives it

        prediction = model.predict(face_input)
        predicted_class = int(np.argmax(prediction))
        os.remove(filepath)

        return jsonify({'emotion': predicted_class})
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=5000)