#
# What is happening here you may ask?
#
# --> It's simple, I'm just overlapping rectangles with the color
# corresponding to the average value of their location on the image.
# This illustrate the problem that often the average color of an image
# is something greying and not very usable for builfing a color LUT
# of different colors...
#

import cv2
import numpy as np
import argparse
from pixelMirrorToolboox import toolboox
import glob
import cPickle
import os
import time

# parameter for the window
cv2.namedWindow("Mirror", cv2.WINDOW_NORMAL)

# grab one frame and get some information about it
camera = cv2.VideoCapture(0)
(grabbed, frame) = camera.read()
shape_image 	 = np.shape(frame)

print "We start the bazard"

# variables for placing a color rectangle in the center of the frame
ptCenter = (shape_image[1]/2,shape_image[0]/2)
vec_w = shape_image[0] / 2 ** np.arange(6,0,-1)	
vec_h = shape_image[1] / 2 ** np.arange(6,0,-1)
pt_1 = (ptCenter[0] - (640 / 2), ptCenter[1] - (480 / 2))
pt_2 = (ptCenter[0] + (640 / 2), ptCenter[1] + (480 / 2))

print vec_w

thickness = -1
counter_number_frames = 50
counter_number_frame_update = 0
counter_hit = len(vec_w) 
counter_hit_update = -1


while True:
	# grab the current frame
	(grabbed, frame) = camera.read()
	frame_original = cv2.flip(frame,1)

	if counter_number_frame_update < counter_number_frames:
		if counter_hit_update > -1:
			if counter_hit_update == 0:
				rectangle_area = frame_original[ptCenter[1] - (vec_h[counter_hit_update] / 2): ptCenter[1] + (vec_w[counter_hit_update] / 2),
			     							    ptCenter[0] - (vec_w[counter_hit_update] / 2): ptCenter[0] + (vec_h[counter_hit_update] / 2),:]
				mean_rectangle = toolboox.describe_mean(rectangle_area)
				color = (mean_rectangle[0], mean_rectangle[1], mean_rectangle[2])	
				pt_1 = (ptCenter[0] - vec_h[counter_hit_update], ptCenter[1] - vec_w[counter_hit_update])
				pt_2 = (ptCenter[0] + vec_h[counter_hit_update], ptCenter[1] + vec_w[counter_hit_update])
				cv2.rectangle(frame_original, pt_1, pt_2, color, thickness)
			else:
				vec = np.arange(counter_hit_update,-1,-1)
				list_color = []
				list_pt1 = []
				list_pt2 = []
				for ii in vec:
					rectangle_area = frame_original[ptCenter[1] - (vec_h[ii] / 2): ptCenter[1] + (vec_w[ii] / 2),
					 							    ptCenter[0] - (vec_w[ii] / 2): ptCenter[0] + (vec_h[ii] / 2),:]
					mean_rectangle = toolboox.describe_mean(rectangle_area)
					color = (mean_rectangle[0], mean_rectangle[1], mean_rectangle[2])
					list_color.append(color)	
					list_pt1.append((ptCenter[0] - vec_h[ii], ptCenter[1] - vec_w[ii]))
					list_pt2.append((ptCenter[0] + vec_h[ii], ptCenter[1] + vec_w[ii]))
	
				for ii in np.arange(0,counter_hit_update):
					cv2.rectangle(frame_original, list_pt1[ii], list_pt2[ii], list_color[ii], thickness)
				
				#print list_color
				del list_color, list_pt1, list_pt2			

	else:
		counter_number_frame_update = 0
		if counter_hit_update < 5:
			counter_hit_update = counter_hit_update + 1
		else:
			counter_hit_update = -1


	# show the frame
	cv2.imshow("Mirror", frame_original)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
	counter_number_frame_update = counter_number_frame_update + 1

camera.release()
cv2.destroyAllWindows()