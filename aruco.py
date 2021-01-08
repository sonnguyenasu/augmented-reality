import cv2
from cv2.aruco import *
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import numpy as np
# fig = plt.figure()
# nx = 4
# ny = 3
# for i in range(1, nx*ny+1):
#     ax = fig.add_subplot(ny,nx, i)
#     img = aruco.drawMarker(aruco_dict,i, 700)
#     plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
#     ax.axis("off")

# plt.savefig("markers.jpeg")
# plt.show()
start = time.time()
frame = cv2.imread('markers.jpeg')
w = 100
h = 100
#src = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
M = np.array([[0.3,-1.2,600],[0.7,-0.3,-80],[0.,0.,1.0]],dtype='float')

#dst = cv2.perspectiveTransform(src,M)
frame = cv2.warpPerspective(frame,M,(1000,1000))
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
print(time.time()-start)
cv2.imshow('f',frame_markers)
cv2.waitKey(0)
