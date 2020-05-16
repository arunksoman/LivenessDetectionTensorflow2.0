"""
################ USAGE ###############
If you are not on Project Directory of FaceLivenessRecognition use following command:
1. cd Projecct


####### For Save Real Faces from Videos(You can use this command first time) ######
2. python dataset_preparation.py --input Videos/Name_of_real_video.avi --output Dataset/Real --skip 4

####### For Save Real Faces from Videos(From seond time onwards) ######
* Go to your Dataset/Real Folder. Then Find the name of last saved file. It might be something like 00265.png
* So name of our next file should be 00266.png. 
    So you have to specify --read flag in order to start from 266th image. For example:
2. python dataset_preparation.py --input Videos/Name_of_real_video.avi --output Dataset/Real --skip 4 --read 35

####### For Save Fake Faces from Videos(You can use this command first time) ######
3. python dataset_preparation.py --input Videos/Name_of_fake_video.avi --output Dataset/Fake --skip 4

####### For Save Fake Faces from Videos(From seond time onwards) ######
4. python dataset_preparation.py --input Videos/Name_of_Fake_video.avi --output Dataset/Fake --skip 4 --read 35
"""

from configurations import *
import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", type=str, required=True,
    help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True,
    help="path to output directory of cropped faces")
ap.add_argument("-s", "--skip", type=int, default=4,
	help="# of frames to skip before applying face detection")
ap.add_argument("-r", "--read", type=int, default=0,
	help="Name of the file from where we have to start")

args = vars(ap.parse_args())
net = cv2.dnn.readNetFromCaffe(PROTXT_PATH, CAFFEMODEL_PATH)

# open a pointer to the video file stream and initialize the total
# number of frames read and saved thus far
vs = cv2.VideoCapture(args["input"])
read = 0
saved = args["read"]
# loop over frames from the video file stream
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

	# grab the frame dimensions and construct a blob from the frame
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	# ensure at least one face was found
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]
		# ensure that the detection with the largest probability also
		# means our minimum probability test (thus helping filter out
		# weak detections)
		if confidence > 0.50:
			# compute the (x, y)-coordinates of the bounding box for
			# the face and extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY, startX:endX]
			# write the frame to disk
			p = os.path.sep.join([args["output"],
				"{}.png".format(str(saved).zfill(5))])
			cv2.imwrite(p, face)
			saved += 1
			print("[INFO] saved {} to disk".format(p))
# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()