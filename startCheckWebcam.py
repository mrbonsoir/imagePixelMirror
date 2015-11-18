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


print 'camerea FPS here %2.4f.' % camera.get(cv2.CAP_PROP_FPS)
print camera.get(cv2.CAP_PROP_FPS)

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