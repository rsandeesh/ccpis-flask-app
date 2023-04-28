#app.py
from flask import Flask, request, jsonify
import json
import os
import urllib.request
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import pandas as pd

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
CLASS_NAME_ARR = ['Fish', 'Flower',  'Gravel',  'Sugar']
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_latest_image():
    root_folder_path = 'runs\detect'
    subfolders = [os.path.join(root_folder_path, folder) for folder in os.listdir(root_folder_path) if os.path.isdir(os.path.join(root_folder_path, folder))]
    subfolders.sort(key=lambda x: os.path.getctime(x), reverse=True)
    latest_subfolder = subfolders[0] if len(subfolders) > 0 else None
    if latest_subfolder is not None:
        files = os.listdir(latest_subfolder)
        files = [os.path.join(latest_subfolder, file) for file in files]
        files = [file for file in files if os.path.isfile(file)]
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_image_path = files[0] if len(files) > 0 else None
    else:
        latest_image_path = None
    return latest_image_path

@app.route('/')
def main():
    return 'Homepage'

#Testing Api to test sending the image with the response
@app.route('/image')
def get_image():
    filename = '\runs\detect\predict100\detect.jpg'
    
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
    result_list = [] #outer array/list
    img_no = 0
    for file in files: 
        patterns = [] # to keep the identified class names  
        json_dict = {} 
        if file and allowed_file(file.filename):
            # Load a model
            loaded_model = YOLO("./model/best.pt")  # load a custom model

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # print("File Path --->",file_path)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # print("------> File Saved SuccessFully")
            
            # print(pred_and_plot(loaded_model, image))
            results = loaded_model(file_path, save=True)  
           
            print("------------- image - {} ----------".format(img_no))
            for result in results:
                p = result.boxes.cls
                isEmpty = p.numel() == 0
                if(isEmpty):
                    patterns.append('None')
                    print('No Patterns in Image - {}'.format(img_no))
                else:
                    print('RECEIVED TENSOR --->', p)
                    integer_val = int(p.item())
                    print('CLASS_INTEGER ---> ',integer_val)
                    patterns.append(CLASS_NAME_ARR[integer_val])
            #setting results to the JSON dict
            json_dict['patterns'] = patterns
            
            filename1 = get_latest_image()
    
            with open(filename1, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                json_dict['base64Img'] = encoded_string
            img_no = img_no +1
            # encoded_image_strings.append(results)
            json_str = json.dumps(json_dict)
            result_list.append(json_str)
        else:
            errors[file.filename] = 'File type is not allowed'
       
    return str(result_list)
 
if __name__ == '__main__':
    app.run(debug=False)


# [{
#     "base64Img":'ksujbdida',
#     "patterns" :['fish', 'flower']
# },
# {

# }]