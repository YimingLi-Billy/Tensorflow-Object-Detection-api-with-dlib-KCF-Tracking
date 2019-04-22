from imutils.video import FPS
import numpy as np
import argparse
import imutils
import dlib
import cv2
import tensorflow as tf
import sys
import time
import os

from utils import label_map_util
from utils import visualization_utils as vis_util


############################################### I N T E R F A C E #######################################################

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
VIDEO_NAME = 'videos/lab_video1.avi'

OUTPUT = 'output/lab_video1_output.avi'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
# PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'TensorFlow-classifiers-master/boxes/frozen_inference_graph.pb')
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
# PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 90

skip_frames = 10
min_score_thresh = 0.5
max_boxes_to_draw = 20

#########################################################################################################################


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)
    
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


vs = cv2.VideoCapture(PATH_TO_VIDEO)

(h, w) = (None, None)

if (vs.isOpened()== False): 
    print("Error opening video stream or file")

writer = None

trackers = []
labels = []

totalFrames = 0

fps = FPS().start()

while True:
	(grabbed, frame) = vs.read()

	if frame is None:
		break

	frame = imutils.resize(frame, width=600)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame_expanded = np.expand_dims(frame, axis=0)

	if OUTPUT is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(OUTPUT, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

	if totalFrames % skip_frames == 0:

		if w is None or h is None:
			(h, w) = frame.shape[:2]

		(boxes, scores, classes, num) = sess.run(
			[detection_boxes, detection_scores, detection_classes, num_detections],
			feed_dict={image_tensor: frame_expanded})
        
		boxes = np.squeeze(boxes)
		classes = np.squeeze(classes).astype(np.int32)
		scores = np.squeeze(scores)
        
		trackers = []
        
		for i in range(min(max_boxes_to_draw, boxes.shape[0])):
			if scores is None or scores[i] > min_score_thresh:
				box = tuple(boxes[i].tolist())
				startY = int((box[0] * h))
				startX = int((box[1] * w))
				endY = int((box[2] * h))
				endX = int((box[3] * w))

				t = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				t.start_track(rgb, rect)

				if classes[i] in category_index.keys():
					class_name = category_index[classes[i]]['name']
				else:
					class_name = 'N/A'

				label = class_name
				labels.append(label)
				trackers.append(t)

				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
				cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	elif len(trackers) != 0:
		#for (t, l) in zip(trackers, labels):
		for t in trackers:
			t.update(rgb)
			pos = t.get_position()

			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)
            
			cv2.putText(frame, label, (startX, startY - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	if writer is not None:
		writer.write(frame)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

	totalFrames += 1
	fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if writer is not None:
	writer.release()

cv2.destroyAllWindows()
vs.release()