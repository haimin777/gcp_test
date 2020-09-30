
import json
import requests

import os
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file

from utils import create_bboxes_array
import utils
import cv2
import numpy as np
from PIL import Image
import string
import matplotlib.pyplot as plt

from celery import Celery
import tools


app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localredis:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localredis:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'webp'])
app.config['UPLOAD_FOLDER'] = '/home/flask/app/web/results'



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@celery.task
def detects_calculation(image):

    # prepare data to send for TensorFlowServing running container
    #data = json.dumps({"signature_name": "serving_default", "instances": png_lung_image.tolist()})
    #image = utils.compute_input(np.array(image))
    data = json.dumps({"signature_name": "serving_default", "instances": image})

    # Make request for prediction
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://ocrmodel:8501/v1/models/tf_models:predict', data=data, headers=headers)
    #json_response = requests.post('http://localhost:8501/v1/models/tf_models:predict', data=data, headers=headers)

    predictions = json.loads(json_response.text)['predictions']


    return predictions


@celery.task
def recognition_task(image, boxes):

    # prepare data to send for TensorFlowServing running container
    #data = json.dumps({"signature_name": "serving_default", "instances": png_lung_image.tolist()})

    X = create_bboxes_array([np.array(image).astype('float32')], [np.array(boxes).astype('float32')])
    data = json.dumps({"signature_name": "serving_default", "instances": X[:,:,:,:1].tolist()})

    # Make request for prediction
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://ocrrecognizer:8501/v1/models/tf_models:predict', data=data, headers=headers)
    #json_response = requests.post('http://localhost:8501/v1/models/tf_models:predict', data=data, headers=headers)

    predictions = json.loads(json_response.text)['predictions']


    return predictions


@app.route('/')
def index():
    return render_template('index.html')

# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    alphabet = string.digits + string.ascii_letters + '$%. ♠♥♦♣'  # '$%. '
    recognizer_alphabet = ''.join(sorted(set(alphabet.lower())))
    blank_label_idx = len(recognizer_alphabet)
    file = request.files['file']
    if file and allowed_file(file.filename):

        filename = file.filename
        #save_filename = filename.split('.')[0]+'.jpg'
        # save received image
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(img_path)

        #img = Image.open(img_path)
        img = tools.read(img_path)
        img = utils.compute_input(img)

        img = np.expand_dims(img, 0)
        # Call TensorFlowServing api for predict

        masks_async = detects_calculation.delay(img.tolist())
        predictions = masks_async.get()

        bboxes = utils.getBoxes([np.array(predictions[0]), ],
                                detection_threshold=0.7,
                                text_threshold=0.4,
                                link_threshold=0.4,
                                size_threshold=10)

        #img_for_recognition = tools.read(img_path) # diferent read algor ????
        img = tools.read(img_path)

        recognized_res = recognition_task.delay(img.tolist(), bboxes[0].tolist())
        res = recognized_res.get()


        res = [''.join([recognizer_alphabet[idx] for idx in row if idx not in [blank_label_idx, -1]])
                        for row in res]
        prediction_groups =  [list(zip(predictions, boxes)) for predictions, boxes in zip([res], bboxes)]
        fig, axs = plt.subplots(nrows=1, figsize=(10, 10))
        tools.drawAnnotations(img, predictions=prediction_groups[0], ax=axs)
        fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'ocr_' + filename))
        return send_from_directory(app.config['UPLOAD_FOLDER'], 'ocr_' + filename)



if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("5000"),
        debug=True)
