from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.event import EventDispatcher
from kivy.uix.screenmanager import (ScreenManager, Screen, NoTransition, SwapTransition) 
from kivy.lang import Builder
import sys
import time
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



class ScreenOne(Screen): 
    pass
   
class ScreenTwo(Screen): 
    pass
  
class ScreenThree(Screen): 
    pass
  
class ScreenFour(Screen): 
    pass
  
class ScreenFive(Screen): 
    pass


# # The ScreenManager controls moving between screens 
screen_manager = ScreenManager() 
   
# # Add the screens to the manager and then supply a name 
# # that is used to switch screens 
screen_manager.add_widget(ScreenOne(name ="screen_one"))
# screen_manager.add_widget(ScreenTwo(name ="screen_two")) 
# screen_manager.add_widget(ScreenThree(name ="screen_three")) 
# screen_manager.add_widget(ScreenFour(name ="screen_four")) 
# screen_manager.add_widget(ScreenFive(name ="screen_five"))

# class MyEventDispatcher(EventDispatcher):
#     def __init__(self, **kwargs):
#         self.register_event_type('on_test')
#         super(MyEventDispatcher, self).__init__(**kwargs)

#     def redirect_me(self, label):
#         # when redirect_me is called, the 'on_test' event will be
#         # dispatched with the label
#         if label == "Real":
#             self.dispatch('on_test', label)

#     def on_test(self, *args):
#         print("I am dispatched", args)

Builder.load_string(""" 
<ScreenOne>: 
    BoxLayout:

""")
class CamApp(App):

    def build(self):
        self.img1=Image()
        layout = BoxLayout()
        layout.add_widget(self.img1)
        #opencv2 stuffs
        self.capture = cv2.VideoCapture('000.avi')
        # cv2.namedWindow("CV2 Image")
        Clock.schedule_interval(self.update, 1.0/33.0)
        return layout

    def update(self, dt):
        # display image from cam in opencv window
        ret, frame = self.capture.read()
        # frame = cv2.flip(frame, -1)
        # cv2.imshow("CV2 Image", frame)
        # convert it to texture
        if frame is None:
            sys.exit()
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

        buf1 = frame.copy()
        buf1 = cv2.flip(buf1, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
        #if working on RASPBERRY PI, use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer. 
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.img1.texture = texture1

if __name__ == '__main__':
    CamApp().run()