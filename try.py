# import the necessary packages
from threading import Thread
import sys
import cv2
# import the Queue class from Python 3
if sys.version_info >= (3, 0):
	from queue import Queue
# otherwise, import the Queue class for Python 2.7
else:
	from Queue import Queue

class FileVideoStream:
	def __init__(self, path, queueSize=128):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		self.stream = cv2.VideoCapture(path)
		self.stopped = False
		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queueSize)

	def start(self):
		# start a thread to read frames from the file video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self
	def update(self):
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				return
			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()
				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.stop()
					return
				# add the frame to the queue
				self.Q.put(frame)
	def read(self):
		# return next frame in the queue
		return self.Q.get()

	def more(self):
		# return True if there are still frames in the queue
		return self.Q.qsize() > 0
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

import numpy as np
import argparse
import time
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
args = vars(ap.parse_args())
# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream(args["video"]).start()
ret, frame = fvs.read()
#cv2.imshow('f',frame)

time.sleep(1.0)
# start the FPS timer