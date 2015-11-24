import cv2
import numpy as np
import argparse
import os
import time

# parameter for the window
cv2.namedWindow("Mirror", cv2.WINDOW_NORMAL)


# grab one frame and get some information about it
camera = cv2.VideoCapture(0)
(grabbed, frame) = camera.read()
shape_image 	 = np.shape(frame)


# here we display some info abou the webcam
list_prop = (cv2.CAP_PROP_POS_MSEC,cv2.CAP_PROP_POS_FRAMES,cv2.CAP_PROP_POS_AVI_RATIO,
cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS,
cv2.CAP_PROP_FOURCC, cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FORMAT,  
cv2.CAP_PROP_MODE, cv2.CAP_PROP_BRIGHTNESS, cv2.CAP_PROP_CONTRAST, 
cv2.CAP_PROP_SATURATION, cv2.CAP_PROP_HUE, cv2.CAP_PROP_GAIN,
cv2.CAP_PROP_EXPOSURE, cv2.CAP_PROP_CONVERT_RGB, 
cv2.CAP_PROP_RECTIFICATION)

for prop in list_prop:
	print camera.get(prop)


print 'camerea FPS here %2.4f.' % camera.get(cv2.CAP_PROP_FPS)

while True:
	#t_n = time.time()

	# grab the current frame
	(grabbed, frame) = camera.read()
	frame_original = cv2.flip(frame,1)
	
	# show the frame
	cv2.imshow("Mirror", frame_original)
	

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

print list_image_name_db

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()