from utilities.video_utilities import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import utilities
import pickle
import time
import cv2
from configurations import *


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
# vs = cv2.VideoCapture(0+cv2.CAP_DSHOW)
# time.sleep(2.0)
vs = cv2.VideoCapture('010.mp4')
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 600 pixels
    _, frame = vs.read()
    if _ != True:
        break
    frame = utilities.resize(frame, width=600)

    # grab the frame dimensions and convert it to a blob
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

            # draw the label and bounding box on the frame
            label = "{}: {:.4f}".format(label, preds[j])
            cv2.putText(frame, label, (startX, startY + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)

    # show the output frame and wait for a key press
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()