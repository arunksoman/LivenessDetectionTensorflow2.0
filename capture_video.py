from configurations import *
import glob
import cv2
from time import perf_counter, sleep

# Performance counter(perf_counter()) usually used for benchmarking our program.
# Using performance counter we can measure time taken by the program to run in seconds.
# We have to make atleast 20sec video. So that we are using performance counter.
start = perf_counter()

# print(dir(os)) # Check whether the file os module getting import from configurations
sleep(4)
cap = cv2.VideoCapture(0)

videos = glob.glob(VIDEO_DIR + "/*.avi")
if len(videos) == 0:
    file_name = 0
else:
    file_name = len(videos)

file_name = str(file_name).zfill(3)
file_name_with_ext = file_name + ".avi"
abspath_to_file = os.path.join(VIDEO_DIR, file_name_with_ext)
# print(abspath_to_file)
# Define the codec and create VideoWriter object 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(abspath_to_file, fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    out.write(frame)
    finish = perf_counter()
    # print(finish - start)
    cv2.imshow("original Video", frame)

    """
        If we will not used cv2.WaitKey() function, there is chances for hanging down system
        due to high memory leverage. So use that function due to this reason. It will take 1 msec.
        So we taken cv2.waitKey(1) in account and taken condition finish - start > 16 in order to get 
        video file to save for 20-25sec.
    """

    cv2.waitKey(1)
    if (finish - start) > 25:
        break
out.release()
cap.release()
cv2.destroyAllWindows()
