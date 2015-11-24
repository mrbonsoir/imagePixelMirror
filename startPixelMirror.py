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
cv2.namedWindow("MirrorDiff", cv2.WINDOW_NORMAL)

# construct the argument parse and parse the arguments
print "We process the input arguments."
ap = argparse.ArgumentParser()
ap.add_argument("-dbs", "--database_size", type = int,
	help = "number of images in the database")
ap.add_argument("-idbm", "--init_database_method", required = True, 
	help = "cam(default) or rand to initialize the db with colored patched")
ap.add_argument("-nr", "--number_rows", type = int, required = True,
	help = "number of cell/super-pixel for the final mosaic, eg 4 x 4 if nr = 4")
ap.add_argument("-dbp", "--database_path", required = True, 
	help = "path to where to store or read the database of images")
ap.add_argument("-nf","--number_frames", type = int,
	help = "the program stops after 1000 frames or the specidied amount given")
ap.add_argument("-nfu","--number_frames_for_update", type = int,
	help = "the program stops after 1000 frames or the specidied amount given")
args = vars(ap.parse_args())
print "You gave me love, thank you."
print args


# Let's process the parameters
if not args.get("database_size", False):
	database_size = 64
	print "Number of images for the database %1.0f." % database_size
else:
	database_size = args["database_size"]
	print "Number of images for the database %1.0f." % database_size

number_rows = args["number_rows"]
print "Number of cells will be row x row %1.0f x %1.0f." % (number_rows, number_rows)

#if not args.get("number_frames", False):
#	number_frames = 5000
#	print "The application will stop after %1.0f frames have been processed" % number_frames
#else:
#	number_frames = args["number_frames"]
#	print "The application will stop after %1.0f frames have been processed" % number_frames

if not args.get("number_frames_for_update", False):
	number_frame_update = 24
	print "The application update the LUT every %1.0f frames" % number_frame_update
else:
	number_frame_update = args["number_frames_for_update"]
	print "The application will stop after %1.0f frames have been processed" % number_frame_update


# check if the path to the database is empty or not
current_dir = os.getcwd()
path_to_image_database = args["database_path"]

if os.path.exists(current_dir+'/'+path_to_image_database):
    print "Yes the database folder already exists."
    files = glob.glob(current_dir+'/'+path_to_image_database+'/*png')
    for f in files:
        os.remove(f)
    print "Now it's clean."
    path_to_image_database = current_dir+'/'+path_to_image_database 
else:
    os.mkdir(current_dir+'/'+path_to_image_database)
    print "The database folder has been created."
    path_to_image_database = current_dir+'/'+path_to_image_database 

print "Everything has been checked, we can start.\n"    

# grab one frame and get some information about it
camera = cv2.VideoCapture(0)
(grabbed, frame) = camera.read()
shape_image 	 = np.shape(frame)

print 'camerea FPS here %2.4f.' % camera.get(cv2.CAP_PROP_FPS)
print camera.get(cv2.CAP_PROP_FPS)

# Here we record the first images that will served as db
if args["init_database_method"] == "cam":
	# we initialize with the first frames directly
	print "You choose the cam stream as initialization."
	toolboox.fun_initialize_image_db(camera, path_to_image_database, database_size)
elif args["init_database_method"] == "gray":
	# we initialize by creating gray patches covering the brightness range from 0 to 255
	print "You choose the gray patches as initialization."
	toolboox.fun_initialize_gray_patches_image_db(camera, path_to_image_database, database_size)
elif args["init_database_method"] == "rand":
	# we initialize by creating random color patches
	print "You choose the colored patches as initialization."
	toolboox.fun_initialize_color_patches_image_db(camera, path_to_image_database, database_size)
else:
	print "Houston..."
	toolboox.fun_initialize_image_db(camera, path_to_image_database, database_size)

# Here we are saving the data for the database
list_image_name_db, metric_image_db = toolboox.fun_create_image_database2(path_to_image_database, number_rows, database_size)

#print list_image_name_db
#print metric_image_db

print "database made of %1.0f images" % len(list_image_name_db)

print "We start the bazard"
counter_number_frames = 0
counter_number_frame_update = 0

# keep looping until spinning!
t0 = time.time()
while True:
	#t_n = time.time()

	# grab the current frame
	(grabbed, frame) = camera.read()
	frame_original = cv2.flip(frame,1)
	
	# save original image
	toolboox.fun_save_image_for_db(frame_original, path_to_image_database+"/","new_frame")

	# test with new image generated for each frame
	frame_mosaic = toolboox.fun_construct_image_from_index_DB2(frame_original, 
														   	path_to_image_database+"/", 
															list_image_name_db, 
															number_rows, 
															metric_image_db)

	#del frame # this because it crashed almost but why... computer is a mystery

	# save the mosaic image
	toolboox.fun_save_image_for_db(frame_mosaic, path_to_image_database+"/","new_mosaic")

	# show the frame
	cv2.imshow("Mirror", frame_mosaic)
	
	# number_frame_upate
	counter_number_frame_update = counter_number_frame_update + 1
	if counter_number_frame_update > number_frame_update:
		##t1 = time.time()
		##print "we update the database because %1.0f frame have been read in %1.3f" % (counter_number_frame_update,
		##																			  t1 -t0)
		counter_number_frame_update = 0

		list_image_name_db, metric_image_db = toolboox.fun_update_image_db(frame_original, 
									 path_to_image_database+"/", 
									 list_image_name_db, 
									 metric_image_db,
									 number_rows)

		##t0 = time.time()

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

	#if counter_number_frames > number_frames:
	#	break
	#counter_number_frames = 1 + counter_number_frames

print list_image_name_db[-10:]
del list_image_name_db
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()