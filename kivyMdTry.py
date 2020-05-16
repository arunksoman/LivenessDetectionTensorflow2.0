from kivymd.app import MDApp
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty
from kivy.lang import Builder
from kivy.uix.screenmanager import (ScreenManager, Screen, NoTransition, 
        SlideTransition, CardTransition, SwapTransition, FadeTransition,
        WipeTransition, FallOutTransition, RiseInTransition)
from kivy.event import EventDispatcher
from utilities.video_utilities import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import sys
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

Builder.load_string("""
<HomeScreen>:
    # canvas.before:
    #     Color:
    #         rgba: 255, 255, 255, 1
    #     Rectangle:
    #         # self here refers to the widget i.e FloatLayout
    #         pos: self.pos
    #         size: self.size
    NavigationLayout:
        MDNavigationDrawer:
            NavigationDrawerIconButton:
                text: "Login"
                on_release:
                    root.manager.transition.direction = 'left' 
                    root.manager.transition.duration = 1
                    root.manager.current = 'LogMeIn'

    AnchorLayout:
        anchor_x: 'center'
        anchor_y: 'center'
        Button:
            size_hint_y: 0.1
            size_hint_x: 0.2
            height: '48dp'
            text: "Login" 
            background_color : 0, 0, 1, 1 
            on_press: 
                # You can define the duration of the change 
                # and the direction of the slide
                root.manager.transition.direction = 'left' 
                root.manager.transition.duration = 1
                root.manager.current = 'LogMeIn'
<LoginScreen>:
    canvas.before:
        Color:
            rgba: 255,255, 255, 1
        Rectangle:
            # self here refers to the widget i.e FloatLayout
            pos: self.pos
            size: self.size
    Button:
        size_hint_y: None
        height: '48dp'
        text: 'Open Camera'
        on_press:
            root.start_cam()
            # root.show_hide(menu)

<UserHome>:
    canvas.before:
        Color:
            rgba: 0, 128, 128, 0.5
        Rectangle:
            # self here refers to the widget i.e FloatLayout
            pos: self.pos
            size: self.size
    BoxLayout:
        Label:
            id: My_Label
            # text: "Hai " + str(app.username)
""")


class OnFaceRecognition(EventDispatcher):
    def __init__(self, **kwargs):
        self.register_event_type('on_recognize')
        super(OnFaceRecognition, self).__init__(**kwargs)

    def goHome(self, label, username):
        # when goHome is called, the 'on_recognize' event will be
        # dispatched with the label
        if label == "Real":
            self.dispatch('on_recognize', label, username)
            # LoginScreen().on_stop()
            app = MDApp.get_running_app()
            app.screen_manager.transition.duration = 4
            # screen_manager.transition = SlideTransition
            app = MDApp.get_running_app()
            app.username = username
            test = MDApp.get_running_app()
            print(test.username)
            app.screen_manager.get_screen('UserHomePage').ids.My_Label.text = "Hai " + app.username
            app.screen_manager.current = "UserHomePage"

    def on_recognize(self, *args):
        print("I am dispatched", args)


def custom_event_callback(label, username, *args):
    print("Hello, I got an event!", args)

class HomeScreen(Screen): 
    pass
class LoginScreen(Image, Screen):
    def start_cam(self, fps=30, **kwargs):
        super(LoginScreen, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture('000.avi')
        Clock.schedule_interval(self.update, 1.0 / fps)
    def update(self, dt):
        ret, frame = self.capture.read()
        # if frame is None:
        #     print("[Info] Frame not Read... Leaving....")
        #     sys.exit()
        if ret:
            frame = utilities.resize(frame, 480, 480)
            # convert it to texture
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
                        ev = OnFaceRecognition()
                        ev.bind(on_recognize=custom_event_callback)
                        username = 'Arun'
                        ev.goHome('Real', username)
                        self.capture.release()
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
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture
    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.capture.release()

class UserHome(Screen):
    def __init__(self, **kwargs):
        print("[Info] I am Here")
        super(UserHome, self).__init__(**kwargs)
        # app = App.get_running_app()
        # print(app.username)
        # print("Welcome {}".format(app.username))
        # self.ids.My_Label.text = "Welcome {}".format(app.username)

class CamApp(MDApp):
    ######## Global Variables ########
    username = None
    screen_manager = ScreenManager(transition = RiseInTransition()) 
    # Add the screens to the manager and then supply a name 
    # that is used to switch screens 
    def build(self):
        CamApp.screen_manager.add_widget(HomeScreen(name ="HomeScreen")) 
        CamApp.screen_manager.add_widget(LoginScreen(name ="LogMeIn"))
        CamApp.screen_manager.add_widget(UserHome(name ="UserHomePage"))
        # self.capture = None
        # self.my_camera = LoginScreen(capture=self.capture, fps=30)
        return CamApp.screen_manager

if __name__ == '__main__':
    CamApp().run()