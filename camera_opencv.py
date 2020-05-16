import os
from utilities.video_utilities import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import utilities
import pickle
import time
import cv2
from configurations import *
from base_camera import BaseCamera
import requests


model_name = "liveness.model"
label_encoder = "le.pickle"

# load our serialized face detector from disk
print("[INFO] loading face detector...")
net = cv2.dnn.readNetFromCaffe(PROTXT_PATH, CAFFEMODEL_PATH)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = load_model(model_name)
le = pickle.loads(open(label_encoder, "rb").read())

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")

class Camera(BaseCamera):
    video_source = '000.avi'

    def __init__(self):
        # if os.environ.get('OPENCV_CAMERA_SOURCE'):
        #     Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, frame = camera.read()
            if frame is None:
                from app import app
                with app.app_context():
                    # return redirect(url_for('ctrl_video'))
                    # from flask import redirect, url_for
                    # redirect = url_for('ctrl_video')
                    msg = {"msg": "success"}
                    r = requests.post("http://127.0.0.1:5000/", json=msg)
                break
            # grab the frame dimensions and convert it to a blob
            frame = utilities.resize(frame, 400)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > 0.85:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face and extract the face ROI
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # ensure the detected bounding box does fall outside the
                    # dimensions of the frame
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    # extract the face ROI and then preproces it in the exact
                    # same manner as our training data
                    face = frame[startY:endY, startX:endX]
                    if len(face) == 0:
                        continue
                    face = cv2.resize(face, (32, 32))
                    face = face.astype("float") / 255.0
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)

                    # pass the face ROI through the trained liveness detector
                    # model to determine if the face is "real" or "fake"
                    preds = model.predict(face)[0]
                    j = np.argmax(preds)
                    label = le.classes_[j]
                    if label == "Real":
                        rect_color = (255, 128, 255)
                    else:
                        rect_color = (0, 0, 255)
                    # draw the label and bounding box on the frame
                    label = "{}: {:.2f}".format(label, preds[j])
                    cv2.rectangle(frame, (startX - 1 , startY - 20), (startX - int((startX - endX) / 1.5), startY), 
                        rect_color, -1)
                    cv2.putText(frame, label, (startX, startY - 5),
                        cv2.FONT_HERSHEY_PLAIN, 0.80, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        rect_color, 2)

            cv2.waitKey(1)
            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame)[1].tobytes()