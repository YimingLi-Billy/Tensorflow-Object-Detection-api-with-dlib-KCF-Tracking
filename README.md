## Tensorflow-Object-Detection-api-with-dlib-KCF-Tracking
This repository contains a program that can run the object detector every few frames and track the bounding boxes with dlib kcf tracker. The purpose is to reduce the computational load and increase the run-speed of the program. 

Basically this program takes in a video and outputs another video that has the bounding boxes drawn on the detected objects. 

## Setup
If the Tensorflow object detection api has already been set up in your device, just simply download the once_a_while.py file and paste it to your directory.

If the Tensorflow object detection api has not been set up, you may use the link in the Download.txt file to download what you need to run the program. You may find the setup step helpful in here: https://github.com/tensorflow/models/tree/master/research/object_detection.

# Libraries
numpy

argparse

imutils

dlib

cv2

tensorflow

You may install them by using pip install in the command prompt.

## Interface
![Interface](https://user-images.githubusercontent.com/46501711/56524595-b3667400-6517-11e9-8de3-caabcd0423a9.JPG)

This is the interface panel of the program. After setting up, you may enter the detection model you want to use, as well as the .ph and .phtxt file. The NUM_CLASSES value depends on the number of labels in the pbtxt file. Next, select the video file you want to process and specify the name for the output video. 

Besides, skip_frames is how frequently you want to run the object detection. 
