import cv2
import argparse
import numpy as np
import math
from utils import *
import time

cap = cv2.VideoCapture(0)
show=False
img = None
track=False
counter = 0 
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=10)   

flann = cv2.BFMatcher()#cv2.FlannBasedMatcher(index_params,search_params)
sift = cv2.SIFT_create()

refPt = []
temp = []
threshold = 350
ret, frame = cap.read()
binary = np.zeros((frame.shape[0],frame.shape[1]))
camera_matrix =  np.array([[519.89300222,   0.,         307.20316512],
 [  0.,         517.02897669, 227.13200295],
 [  0.,           0.,          1.        ]])

test_text = cv2.imread('text.png')
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt,temp
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt.append((x, y))
		temp.append((x,y))
		

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		# draw a rectangle around the region of interest
		cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("frame", frame)
	elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
		try:
			temp[1] = (x,y)
		except:
			temp.append((x,y))
dst = []
while cap.isOpened():
    ret, frame = cap.read()
    counter = (counter + 1) % 3
    try:
        fps = 1/(end-start)
    except: fps = 0
    start = time.time()
    if ret:
        #flip the frame coming from webcam
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(0,0),fx=0.2,fy=0.2)
        gray = cv2.resize(gray,(frame.shape[1],frame.shape[0]))
        #set callback to control the mouse
        cv2.setMouseCallback("frame", click_and_crop)
        k = cv2.waitKey(10)&0xFF
        #end= time.time() ~50fps
        #show the temporary chosen rectangle 
        if len(temp) == 2:
            cv2.rectangle(frame, temp[0], temp[1], (0, 255, 0), 2)
        
        #tracking the template with sift features
        if track and counter == 1:
            
            #print(len(kp2))
            
            #n,lbls = keypoint_clustering([kp2[m.trainIdx] for m,_ in matches_])
            n = 1
            
            for i in range(n):
                kp2, des2 = sift.detectAndCompute(gray,None)
                matches_ = flann.knnMatch(des1,des2,k=2)
                matches = matches_#np.array(matches_)[lbls==i].tolist()
                matches = sorted(matches,key=lambda x: x[0].distance)
                matches = [match for match in matches if match[0].distance<threshold]
                if len(matches) > max(0.5*len(matches_)/n,20) and matches[0][0].distance < 200:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m,_ in matches[:]]).reshape(-1,1,2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m,_ in matches[:]]).reshape(-1,1,2)
                    M,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
                    h,w = img.shape
                    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                    dsts = cv2.perspectiveTransform(pts,M)
                    if len(dst) == 0: dst.append(dsts)
                    elif len(dst) == 1:
                        if np.sum((dst[0]-dsts)**2) < 400:
                            continue
                        else:
                            dst[0] = dsts
                    
                    cv2.fillPoly(gray,[np.int32(dst[-1])],255)
            
        #end= time.time()  ~5fps  
        try:
            for d in dst:
                frame = cv2.polylines(frame, [np.int32(d)], True, 255, 3, cv2.LINE_AA)
        except:
            pass        
        cv2.imshow('frame',frame)
        if k == ord('q'):
            break
        
        #press d to start tracking
        elif k == ord('d'):
            kp1, des1 = sift.detectAndCompute(img,None)
            track = True

        elif k == ord('z'):
            #reset all items
            temp = []
            track = False
            img = None
        
        #press s to crop the region of interest
        #mah, just combine s and d then
        elif k == ord('s'):
            #save the cropped part out and save it into img variable
            img = gray[temp[0][1]:temp[1][1],temp[0][0]:temp[1][0]]
            
            test_text = cv2.resize(test_text,(img.shape[1],img.shape[0]))
            temp = []
            kp1, des1 = sift.detectAndCompute(img,None)
            track = True
        if img is not None:
            cv2.imshow('img',img)
        end = time.time()
    else: break
