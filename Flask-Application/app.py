import os
import numpy as np
import cv2
from flask import Flask, request, render_template, send_from_directory
from keras.models import load_model
from keras.preprocessing.image import img_to_array


from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
CATEGORIES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
IMG_SIZE = 224
UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = load_model('./model/model.h5')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare(filepath):
    img_array = cv2.imread(filepath)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img_array) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict(file):
    prediction = model.predict(prepare(file))
    return CATEGORIES[np.argmax(prediction)]


@app.route("/")
def home():
    return render_template('home.html', label='', imagesource='')


@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template("home.html", label='No file uploaded', imagesource='')

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        output = predict(file_path)
        return render_template("home.html", label=output, imagesource=file_path)

    return render_template("home.html", label='Invalid file type', imagesource='')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True)
