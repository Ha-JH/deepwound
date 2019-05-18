from flask import Flask, jsonify, request
import numpy as np
from ensemble import *
import os
from keras.applications import VGG16
from keras.models import load_model, Sequential, Model
from keras.layers import Dropout, Dense, Flatten
from scipy.misc import imread, imresize
import datetime

app = Flask(__name__)

@app.route('/')
def homepage():
    return "Hello!"

@app.route('/analyze', methods=['POST'])
def predict():
    start = datetime.datetime.now()
    file = request.files["image"]
    file.save("test.jpg")
    files = ["vgg16-6-FINAL.h5", "vgg16-10-FINAL.h5", "vgg16-14-FINAL.h5"]
    ens = Ensemble(files)
    res = ens.predict(imresize(imread("test.jpg"), (224, 224)))
    end = datetime.datetime.now()
    delta = end - start
    time = delta.total_seconds() * 1000
    return jsonify({"time": time, "results": {"wound": int(res[0]), "infection": int(res[1]), "gran_tissue": int(res[2]), "fibri_exu": int(res[3]), "open": int(res[4]), "drainage": int(res[5]), "steri_strips": int(res[6]), "staples": int(res[7]), "sutures": int(res[8])}})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
