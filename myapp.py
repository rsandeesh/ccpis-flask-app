# app.py
from flask import Flask, request, jsonify , send_file
import json
import os
import urllib.request
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import pandas as pd
import cv2
import torch
from flask_cors import CORS, cross_origin
# from google.colab.patches import cv2_imshow

import base64
 
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
 
app.secret_key = "caircocoders-ednalan"
 
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route('/')
def main():
    return 'Homepage'

#Testing Api to test sending the image with the response
@app.route('/image')
def get_image():
    filename = 'runs\detect\predict\dbd3c.jpg'
    
    with open(filename, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    return str(encoded_string)
 
@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    # check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
 
    files = request.files.getlist('files[]')
     
    errors = {}
    success = False
    response={}
    for file in files:      
        if file and allowed_file(file.filename):
            # Load a model
            loaded_model = YOLO("./model/best.pt")  # load a custom model

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            print("File Path --->",file_path)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("------> File Saved SuccessFully")
            
            # print(pred_and_plot(loaded_model, image))
            results = loaded_model(file_path, save=True)  
            print("Model Results  --->",results)
            # results_dict = results.to_dict()
            # results_json = json.dumps(results)
            # print(results.join(map(str, list)))
            # for i in range(0, len(results)):
            #  print(i)
            # print(results.__getitem__('orig_img'))
            for result in results:
                response={
                    'data':result.orig_img
                }
            
            filename = 'runs\detect\predict\dbd3c.jpg'
    
            with open(filename, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
        else:
            errors[file.filename] = 'File type is not allowed'
        return str(encoded_string)

if __name__ == '__main__':
    app.run(debug=False)