# coding: utf-8
 
import numpy as np
import cv2
import shutil
import os
 
print ("the opencv version: {}".format(cv2.__version__))
 
save_path = "./save_each_frames_front"
if(os.path.exists(save_path)):
    shutil.rmtree(save_path)
os.mkdir(save_path)
 
#cap = cv2.VideoCapture("./trailer.mp4")
cap = cv2.VideoCapture()
cap.open("T-21026633_FRONT_DAB_85.avi")
if cap.isOpened() != True:
    os._exit(-1)
 
#get the total numbers of frame
totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print ("the number of frames is {}".format(totalFrameNumber))
 
#set the start frame to read the video
frameToStart = 1
cap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart);
 
#get the frame rate
rate = cap.get(cv2.CAP_PROP_FPS)
print ("the frame rate is {} fps".format(rate))
 
# get each frames and save
frame_num = 0
while True:
    ret, frame = cap.read()
    if ret != True:
        break
    img_path = os.path.join(save_path, str(frame_num)+".jpg")
    cv2.imwrite(img_path,frame)
    frame_num = frame_num + 1
 
    # wait 10 ms and if get 'q' from keyboard  break the circle
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
 
cap.release()