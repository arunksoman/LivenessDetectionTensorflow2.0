from importlib import import_module
import os
from waitress import serve
from flask_cors import CORS
from flask import Flask, render_template, Response, url_for, redirect, jsonify, session,request
from flask_sqlalchemy import SQLAlchemy
# import threading
from time import sleep
from WebGui.webgui import FlaskUI #get the FlaskUI class

# import camera driver
"""
# if os.environ.get('CAMERA'):
#     Camera = import_module('camera_' + os.environ['CAMERA']).Camera
# else:
#     from cam import Camera
"""
from camera_opencv import Camera
# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera

app = Flask(__name__)
ui = FlaskUI(app)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + BASE_DIR + os.path.sep + "db.sqlite"
app.config['SECRET_KEY'] = "hahahahahhahahhahhahahhahahhchi"
app.config['SERVER_NAME'] = "127.0.0.1:5000"
db = SQLAlchemy(app)

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=["POST", "GET"])
def index():
    """Video streaming home page."""
    if request.method == "POST" and request.get_json()['msg'] == "success":
        return render_template("test.html")
    return render_template("videofeed.html")
    # return redirect("admin")

@app.route('/video_feed')
def video_feed():
    frame = gen(Camera())
    # print(Camera.motionStatus)
    return Response(frame, mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/control", methods=["GET", "POST"])
def ctrl_video():
    return render_template('test.html')


if __name__ == '__main__':
    app.run(debug=True)
    # serve(app)
    # ui.run()