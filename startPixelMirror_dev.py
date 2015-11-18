# The idea is to create a video mirror where all the super pixelis 
# would be tiny images continuously recorded.

import cv2
import numpy as np
import argparse
#from pixelMirrorToolboox.rgbhistogram import RGBHistogram
from pixelMirrorToolboox import toolboox
import glob
import cPickle
import os
import time
#from pixelMirrorToolboox import searcher

# parameter for the window
cv2.namedWindow("Mirror", cv2.WINDOW_NORMAL)
cv2.namedWindow("MirroirDatabase", cv2.WINDOW_NORMAL)

# construct the argument parse and parse the arguments
print "We process the input arguments."
ap = argparse.ArgumentParser()
ap.add_argument("-dbs", "--database_size", type = int,
	help = "number of images in the database")
ap.add_argument("-nr", "--number_rows", type = int, #required = True,
	help = "number of cell/super-pixel for the final mosaic, eg 4 x 4 if nr = 4")
ap.add_argument("-dbp", "--database_path", #required = True, 
	help = "path to where to store or read the database of images")
ap.add_argument("-nf","--number_frames", type = int,
	help = "the program stops after 1000 frames or the specidied amount given")
ap.add_argument("-nfu","--number_frames_for_update", type = int,
	help = "the program stops after 1000 frames or the specidied amount given")
args = vars(ap.parse_args())
print args

# Let's process the parameters
#if not args.get("database_size", False):
#	database_size = 64
#	print "Number of images for the database %1.0f." % database_size
#else:#
#	database_size = args["database_size"]
#	print "Number of images for the database %1.0f." % database_size

#number_rows = args["number_rows"]
#print "Number of cells will be row x row %1.0f x %1.0f." % (number_rows, number_rows)

#if not args.get("number_frames", False):
#	number_frames = 5000
#	print "The application will stop after %1.0f frames have been processed" % number_frames
#else:
#	number_frames = args["number_frames"]
#	print "The application will stop after %1.0f frames have been processed" % number_frames

#if not args.get("number_frames_for_update", False):
#	number_frame_update = 24
#	print "The application update the LUT every %1.0f frames" % number_frame_update
#else:
#	number_frame_update = args["number_frames_for_update"]
#	print "The application will stop after %1.0f frames have been processed" % number_frame_update


# check if the path to the database is empty or not
#current_dir = os.getcwd()
#path_to_image_database = args["database_path"]

#if os.path.exists(current_dir+'/'+path_to_image_database):
#    print "Yes the database folder already exists."
#    files = glob.glob(current_dir+'/'+path_to_image_database+'/*png')
#    for f in files:
#        os.remove(f)
#    print "Now it's clean."
#    path_to_image_database = current_dir+'/'+path_to_image_database 
#else:
#    os.mkdir(current_dir+'/'+path_to_image_database)
#    print "The database folder has been created."
#    path_to_image_database = current_dir+'/'+path_to_image_database 

#print "Everything has been checked, we can start.\n"    

# grab one frame and get some information about it
camera = cv2.VideoCapture(0)
(grabbed, frame) = camera.read()
shape_image 	 = np.shape(frame)

print 'camerea FPS here %2.4f.' % camera.get(cv2.CAP_PROP_FPS)
print camera.get(cv2.CAP_PROP_FPS)
print "We start the bazard"
counter_number_frames = 0
counter_number_frame_update = 0

# keep looping until spinning!
t0 = time.time()

# display the rectangle in the center of the image
ptCenter = (shape_image[1]/2,shape_image[0]/2)
print ptCenter
scale_factor_for_rectangle1 = 2
pt11 = (ptCenter[0] - (320 / 2), ptCenter[1] - (240 / 2))
pt12 = (ptCenter[0] + (320 / 2), ptCenter[1] + (240 / 2))
pt21 = (ptCenter[0] - (160 / 2), ptCenter[1] - (120 / 2))
pt22 = (ptCenter[0] + (160 / 2), ptCenter[1] + (120 / 2))
pt31 = (ptCenter[0] - (80 / 2), ptCenter[1] - (60 / 2))
pt32 = (ptCenter[0] + (80 / 2), ptCenter[1] + (60 / 2))
pt41 = (ptCenter[0] - (40 / 2), ptCenter[1] - (30 / 2))
pt42 = (ptCenter[0] + (40 / 2), ptCenter[1] + (30 / 2))


number_frame_for_lut = 0#database_size


while True:
	# grab the current frame
	(grabbed, frame) = camera.read()
	frame_original = cv2.flip(frame,1)
	ss = np.shape(frame)

	rectangle_area1 = frame_original[ptCenter[1] - (320 / 2): ptCenter[1] + (320 / 2),
									ptCenter[0] - (240 / 2): ptCenter[0] + (240 / 2),:]
	rectangle_area2 = frame_original[ptCenter[1] - (160 / 2): ptCenter[1] + (160 / 2),
									ptCenter[0] - (120 / 2): ptCenter[0] + (120 / 2),:]
	rectangle_area3 = frame_original[ptCenter[1] - (80 / 2): ptCenter[1] + (80 / 2),
									ptCenter[0] - (60 / 2): ptCenter[0] + (60 / 2),:]	
	rectangle_area4 = frame_original[ptCenter[1] - (40 / 2): ptCenter[1] + (40 / 2),
									ptCenter[0] - (30 / 2): ptCenter[0] + (30 / 2),:]	
   	mean_rectangle1 = toolboox.describe_mean(rectangle_area1)
	mean_rectangle2 = toolboox.describe_mean(rectangle_area2)
	mean_rectangle3 = toolboox.describe_mean(rectangle_area3)
	mean_rectangle4 = toolboox.describe_mean(rectangle_area4)
	
	color1 = (mean_rectangle1[0], mean_rectangle1[1], mean_rectangle1[2])
	color2 = (mean_rectangle2[0], mean_rectangle2[1], mean_rectangle2[2])
	color3 = (mean_rectangle3[0], mean_rectangle3[1], mean_rectangle3[2])
	color4 = (mean_rectangle4[0], mean_rectangle4[1], mean_rectangle4[2])
	thickness = -1

	cv2.rectangle(frame_original, pt11, pt12, color1, thickness)
	cv2.rectangle(frame_original, pt21, pt22, color2, thickness)
	cv2.rectangle(frame_original, pt31, pt32, color3, thickness)
	cv2.rectangle(frame_original, pt41, pt42, color4, thickness)

	# show the frame
	cv2.imshow("Mirror", frame_original)


	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()