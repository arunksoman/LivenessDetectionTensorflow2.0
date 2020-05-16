1. Create folder named FaceLiveness Recognition any drive you want.
2. Open VSCode. Then go to File $\rarr$ Open folder $\rarr$ navigate to your folder $\rarr$ open folder
3. Create folder structure (The folder structure can be change time to time as per requirements.) as I given below except <span style="color: green">**venv**</span>

```
FaceLivenessRecognition
├─ Project
│  ├─ configurations.py
│  ├─ ConvNet
│  │  ├─ convnet.py
│  │  └─ __init__.py
│  ├─ Dataset
│  │  ├─ Fake
│  │  └─ Real
│  ├─ dataset_preparation.py
│  ├─ docs.md
│  ├─ readme.md
│  ├─ recognize.py
│  ├─ SSDModels
│  │  ├─ deploy.prototxt
│  │  └─ res10_300x300_ssd_iter_140000.caffemodel
│  ├─ train.py
│  ├─ utilities
│  │  ├─ convenience.py
│  │  ├─ paths.py
│  │  ├─ video_utilities
│  │  │  ├─ filestream.py
│  │  │  ├─ streamfromcam.py
│  │  │  ├─ videostream.py
│  │  │  └─ __init__.py
│  │  └─ __init__.py
│  ├─ Video
│  └─ Videos
├─ readme.md
├─ requirements.txt
└─ venv

```
4. Go to terminal $\rarr$ New terminal on your vs code. it will open up an integrated terminal on the bottom.
5. Install virtual environment using following command on integrated terminal. It will take upto 2 or 3 minutes  to execute so wait until that. After executing this command you can see a <span style="color: green">**venv**</span> folder:
 
```python -m venv venv```

6. Then activate your venv using following command:

```.\venv\Scripts\activate```

7. Install necessary python index packages using following command:

```pip install -r requirements.txt```

8. in order to check everything works perfectly. Type **python** on your integrated terminal. It will show python interpreter. Then type following lines and hit enter after each line to execute those lines.

```python
import tensorflow as tf
import keras
```

If these things not throwing any error our virtual environment is ready for doing this project. If this throws some error like <span style="color: red">**DLL file missing**</span> please contact me.

## Dataset Preparation

### Step 1
1. Run **capture_video.py** file. After running this file you can see your face on a window. You have to move your face back and forth as well as change your angle of face infront of camera. it will save your video on the folder named **Videos in avi format**.
2. Copy that video to your mobile phone. Play that video before your camera while running same program(**capture_video.py**). Do not forgot to <span style="color:red">**write/ notedown names of those files.**</span>
3. Now the video played from your phone is a kind of fake image since your live appearance is not before your camera. So Now we have to create our dataset using these steps.
4. For more accuracy you can use different camera sensors (say camera of your mobile phone) in order to capture video. Then you have to store those videos on your folder Videos(notedown this video as real). Then Play this video on phone and capture that video using **capture_video.py** script. 
5. **Another ways to improve accuracy is you have to choose diffrent lighting conditions for taking those videos as well as people with diffrent skin tones. We are using a shallow convolutional neural network, so our each class(Real and Fake) can have upto 8k-12k images.**

#### program Explanation

```python
from configurations import *
import glob
import cv2
from time import perf_counter
```
These line imports necessary packages. We written configuration.py inorder to maintain project structure tidy and clean as well as store path to necessary pretrained model. Glob is a python package used for find out files using some patterns or wild card entry. For example ***.png** is a pattern which selects all png files with in the specified path and returns those path to those files as a list. Another example of glob is **c??.py**. This indicates glob selects a letter starts with c and that has only 3 letters, also the letter 2 or letter 3 can be anything for example, cup, cap, cat etc.

```python
start = perf_counter()
```

Performance counter(perf_counter()) usually used for benchmarking our program. Using performance counter we can measure time taken by the program to run in seconds. We have to make atleast 20sec video. So that we are using performance counter.

```python
cap = cv2.VideoCapture(0)
```
It will create an instance of your camera. The 0 given indicates it will open up your default camera of device. 

```python
videos = glob.glob(VIDEO_DIR + "/*.avi")
if len(videos) == 0:
    file_name = 0
else:
    file_name = len(videos)
```

The videos variable holds a list of avi files with in the folder Video. If length of list video is zero, the name of newly captured video should be 000.avi otherwise the name should be 00$x$.avi.

Where $x\rarr$ length of list videos.

```python
file_name = str(file_name).zfill(3)
file_name_with_ext = file_name + ".avi"
abspath_to_file = os.path.join(VIDEO_DIR, file_name_with_ext)
# print(abspath_to_file)
# Define the codec and create VideoWriter object 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(abspath_to_file, fourcc, 20.0, (640,480))
```

The string handling function **zfill** is used for prepend 3 zeros infront of file_name which is simply length of list named videos. The forcc is a 4-bit video codec. In order to know more go to [opencv-tutorial](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html) section saving video. In videoWriter element first argument is absolute path to file which we are created with os.path.join with file extension. 20.0 in the cv2.videoWriter() function is 20 frames per second as well as (640, 480) is the height and width of frame.

```python
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    out.write(frame)
    finish = perf_counter()
    # print(finish - start)
    cv2.imshow("original Video", frame)
    cv2.waitKey(1)
    if (finish - start) > 16:
        break
out.release()
cap.release()
cv2.destroyAllWindows()
```
In order to know more about this part go to [opencv-tutorial](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html) section saving video. Also go through comments I written in program.

### step 2

Run dataset_preparation.py file in the Project directory. In order to run this file, read the comments or docstring given above that file. The --skip flag in the program is used for skip the video frame. We are expecting upto 170-280 images from a video of 20-30sec.

```python
from configurations import *
import numpy as np
import cv2
import argparse
```

These lines import necessary packages for us. Here **argparse** is used for parse command line arguments. You might used something like

```java
public class AddTwoNumbers {
   public static void main(String[] args) {
        // your program goes here
   }
}
```
in Java. Here there is an argument args in main function which indiccates java can accept command line argument. If you don't used that yet do not bother about that. Here we imported configuration mainly because in order to access the caffemodel as well as from deploy.protxt file. The caffemodel is a pretrained DNN (Deep Neural Network) model. The every DNN model contains weights and bias to determine the classes. Here we have two classes image with faces and with out faces. The prototxt file contains how the architecture works, how many layers it have, what is the activation function used, is there exists any pooling layers etc. I will share docs about what these terms really means. **Here we uses the SSD(Single shot dtection) model for detect face quickly from video frames**. The SSD model called so, since it skips other layers when pattern recognized in the layers in between and jumps to output layer right from there. It will saves lots of time. That is why we are using this model specifically. 

```python
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", type=str, required=True,
    help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True,
    help="path to output directory of cropped faces")
ap.add_argument("-s", "--skip", type=int, default=16,
	help="# of frames to skip before applying face detection")
ap.add_argument("-r", "--read", type=int, default=0,
	help="Name of the file from where we have to start")

args = vars(ap.parse_args())
```

Here ```ap``` is object of argument parser. I believes it is kind of dictionary. We have to add some keys to that dictionary by using ap.add_argument(). We can specify type of agument(int, float, str) or default values or whether it is required to run the program. You have to read the usage(given as docstring of program) carefully, you might get this. Last line in this section gives a dict with keys as our dashes or hiphens stripped from second option of added argument (Like ```args["skip"] = 16```).

```python
net = cv2.dnn.readNetFromCaffe(PROTXT_PATH, CAFFEMODEL_PATH)
vs = cv2.VideoCapture(args["input"])
read = 0
saved = args["read"]
```
 The first statement loads our caffemodel and corresponding prototxt file. from opencv 3.3.3 onwards it supports pretrained **torch, caffe as well as tensorflow** model to load. vs reads the video file from given path we specified in our command line argument. Once we ran the program we have to check the image files stored on dataset/real or Dataset/Fake. Then look for what was the name of last file saved. It might be something like 00265.png. Now in order to save next image as 00266.png you have to use **--read 266** flag in the command line while running program. The saved variable is responsible for changing the name for each image. So I just assigned that flag to this variable.

```python
while True:
	# grab the frame from the file
	(grabbed, frame) = vs.read()
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
	# increment the total number of frames read thus far
	read += 1
	# check to see if we should process this frame
	if read % args["skip"] != 0:
		continue
```
in order to skip a later parts of loop, we are using --skip flag. While each frame reads our read variable increments. If we want every 4th frame to be processed in order to reduce the amount of data we want to train we can set **--skip 4** as flag. I expect more than 200 images from ~25 sec video. The default value of --skip is 16. You have to reduce that to 2 or 4 in order to get more images(~200) via command line.

```python
	# grab the frame dimensions and construct a blob from the frame
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
```

The frame.shape will give you a tuple (height, width, number_of_color_channels). The number of color channels is unnecessary for us to reconstruct the blob frame. **Blob stand for Binary Large OBjects**. Well it is used to represent a group of pixels having similar values for intensity but different from the ones surrounding it. In the context of deep learning, a blob is also the entire image pre-processed and prepared for classification/training. Such pre-processing usually entails mean subtraction and scaling. Here our caffemodel expects 300 $\times$ 300 image. So we resized our frame to that. The 1.0 (mean substraction) given in the function is scale factor, next (300, 300) is spatial size of image and (104.0, 177.0, 123.0) tuple is also used for  substraction. I will share docs related to **mean substraction**. Now you have to only know it is used for mean substraction.

Then we set our blob as an input to neural network. I told earlier this is a pretrained model, so that we do not want to retrain our face detector or initialize weights or do back propagation or anything. We just want to propagate our network forward with proper input and just want to wait for output. It is what ```detections = net.forward()``` function do.
