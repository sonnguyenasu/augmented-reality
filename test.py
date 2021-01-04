import cv2
import argparse
import numpy as np
import math
from utils import *

cap = cv2.VideoCapture(0)
show=False
img = None
track=False
counter = 0 
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   

flann = cv2.FlannBasedMatcher(index_params,search_params)
orb = cv2.SIFT_create()
refPt = []
temp = []
threshold = 350
bandwidth=None
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

while cap.isOpened():
    ret, frame = cap.read()
    counter = (counter + 1) % 7
    if ret:
        #flip the frame coming from webcam
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #set callback to control the mouse
        cv2.setMouseCallback("frame", click_and_crop)
        k = cv2.waitKey(10)&0xFF
        
        if len(temp) == 2:
            cv2.rectangle(frame, temp[0], temp[1], (0, 255, 0), 2)
        if track:
            matchExist = True
            while matchExist:
                matchExist=False
                kp2, des2 = orb.detectAndCompute(gray,None)
                

                matches_ = flann.knnMatch(des1,des2,k=2)
                n,lbls = keypoint_clustering([kp2[m.trainIdx] for m,_ in matches_])
                for i in range(n):
                    matches = np.array(matches_)[lbls==i].tolist()
                    matches = sorted(matches,key=lambda x: x[0].distance)
                    matches = [match for match in matches if match[0].distance<threshold]
                    #matches = matches[:20]
                    if len(matches) > (0.75*len(matches_)/n):
                        if True:
                            src_pts = np.float32([kp1[m.queryIdx].pt for m,_ in matches[:50]]).reshape(-1,1,2)
                            dst_pts = np.float32([kp2[m.trainIdx].pt for m,_ in matches[:50]]).reshape(-1,1,2)
                            M,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
                            h,w = img.shape
                            pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                            dst = cv2.perspectiveTransform(pts,M)
                            
                            projection = projection_matrix(camera_matrix,M)
                            #rendered = cv2.warpPerspective(test_text,M,(frame.shape[1],frame.shape[0]))
                            #cv2.fillPoly(frame,[np.int32(dst)],(0,0,0))
                            #cv2.imshow('rendered',rendered)
                            #frame += rendered

                            #frame = draw_axis(frame,projection[:,:3],projection[:,3],camera_matrix)
                            frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                            cv2.fillPoly(gray,[np.int32(dst)],255)
                        else:#except Exception as e:
                            print(e)
                    else: matchExist = False
        cv2.imshow('frame',frame)
        
        if k == ord('q'):
            break
        
        #press d to start tracking
        elif k == ord('d'):
            kp1, des1 = orb.detectAndCompute(img,None)
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
            kp1, des1 = orb.detectAndCompute(img,None)
            track = True
        if img is not None:
            cv2.imshow('img',img)
        
    else: break
