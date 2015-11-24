import numpy as np
import cv2
#from pixelMirrorToolboox.rgbhistogram import RGBHistogram
import glob
import toolboox
import time
import shutil
import os

# global parameters
frame_wdth = 480
frame_height = 640 

def fun_initialize_image_db(cam, path_database, database_size):
	'''
	The function saves the first "database_size" frame for to initialize the 
	image database. 
	'''
	counter =  0

	while True: 
		# grab image
		(grabbed, frame) = cam.read()
		frame = cv2.flip(frame,1)

		frame_shuffle = np.zeros(np.shape(frame))
		##	
		##if np.random.randint(2) == 0:
		##	frame_shuffle = frame
		##else:
		##	frame_shuffle = frame[:,:,np.random.permutation(3)]
		frame_shuffle = frame

		# resize frame before saving
		cv2.imwrite(path_database+"/frame_640_480_"+str(counter).zfill(3)+".png", frame_shuffle)

		# resize frame before saving
		frame_db = cv2.resize(frame_shuffle, (320, 240)) # --> 2 x 2 cells
		cv2.imwrite(path_database+"/frame_320_240_"+str(counter).zfill(3)+".png", frame_db)

		frame_db = cv2.resize(frame_shuffle, (160, 120)) # --> 4 x 4 cells
		cv2.imwrite(path_database+"/frame_160_120_"+str(counter).zfill(3)+".png", frame_db)

		frame_db = cv2.resize(frame_shuffle, (80, 60)) # --> 8 x 8 cells
		cv2.imwrite(path_database+"/frame_080_060_"+str(counter).zfill(3)+".png", frame_db)

		frame_db = cv2.resize(frame_shuffle, (40, 30)) # --> 8 x 8 cells
		cv2.imwrite(path_database+"/frame_040_030_"+str(counter).zfill(3)+".png", frame_db)

		frame_db = cv2.resize(frame_shuffle, (20, 15)) # --> 8 x 8 cells
		cv2.imwrite(path_database+"/frame_020_015_"+str(counter).zfill(3)+".png", frame_db)

		# display frame
		cv2.imshow("Mirror", frame)
		counter = counter +1 
		
		if counter >= database_size:
			print "frame database almost initilized"
			break


def fun_initialize_image_db2(cam, path_database, database_size):
	'''
	The function saves the first "database_size" frame for to initialize the 
	image database. 
	'''
	counter =  0


	ptCenter = np.array([320, 240])
	while True: 

		# grab image
		(grabbed, frame) = cam.read()
		frame = cv2.flip(frame,1)
		ss = np.shape(frame)
		# resize frame before saving
		#frame_db = cv2.resize(frame, (320, 240)) # --> 2 x 2 cells
		frame_db = frame[ptCenter[1] - ss[0]/2: ptCenter[1] + 3 * ss[0]/2,   
					     ptCenter[0] - ss[1]/2: ptCenter[0] + 3 * ss[1]/2,:]
		cv2.imwrite(path_database+"/frame_320_240_"+str(counter).zfill(3)+".png", frame_db)
		del frame_db

		#frame_db = cv2.resize(frame, (160, 120)) # --> 4 x 4 cells
		frame_db = frame[ptCenter[1] - ss[0]/8: ptCenter[1] + ss[0]/8,
						  ptCenter[0] - ss[1]/8: ptCenter[0] + ss[1]/8,:]
		cv2.imwrite(path_database+"/frame_160_120_"+str(counter).zfill(3)+".png", frame_db)
		del frame_db

		#frame_db = cv2.resize(frame, (80, 60)) # --> 8 x 8 cells
		frame_db = frame[ptCenter[1] - ss[0]/16: ptCenter[1] + ss[0]/16,
						  ptCenter[0] - ss[1]/16: ptCenter[0] + ss[1]/16,:]
		cv2.imwrite(path_database+"/frame_080_060_"+str(counter).zfill(3)+".png", frame_db)
		del frame_db

		#frame_db = cv2.resize(frame, (40, 30)) # --> 16 x 16 cells
		frame_db = frame[ptCenter[1] - ss[0]/32: ptCenter[1] + ss[0]/32,
						  ptCenter[0] - ss[1]/32: ptCenter[0] + ss[1]/32,:]
		cv2.imwrite(path_database+"/frame_040_030_"+str(counter).zfill(3)+".png", frame_db)
		del frame_db

		frame_db = cv2.resize(frame, (20, 15)) # --> 32 x 32 cells
		frame_db = frame[ptCenter[1] - ss[0]/64: ptCenter[1] + ss[0]/64,
						 ptCenter[0] - ss[1]/64: ptCenter[0] + ss[1]/64,:]
		cv2.imwrite(path_database+"/frame_020_015_"+str(counter).zfill(3)+".png", frame_db)
		del frame_db

		# display frame
		cv2.imshow("Mirror", frame)
		counter = counter +1 
		
		if counter >= database_size:
			print "frame database almost initilized"
			break

def fun_initialize_color_patches_image_db(cam, path_database, database_size):
	'''
	The function saves the first "database_size" frame for to initialize the 
	image database. 
	'''

	# take one frame
	(grabbed, frame) = cam.read()
	frame = cv2.flip(frame,1)

	# create the random values
	data_lut = np.random.randint(0,255,(3,database_size))

	for vec in np.arange(np.shape(data_lut)[1]):
		image_temp = np.ones(np.shape(frame), dtype=float)
		image_temp[:,:,0] = image_temp[:,:,0] * data_lut[0,vec] 
		image_temp[:,:,1] = image_temp[:,:,1] * data_lut[1,vec] 
		image_temp[:,:,2] = image_temp[:,:,2] * data_lut[2,vec]

		cv2.imwrite(path_database+"/frame_640_480_"+str(vec).zfill(3)+".png", image_temp)

		image_temp = cv2.resize(image_temp, (320, 240)) 
		cv2.imwrite(path_database+"/frame_320_240_"+str(vec).zfill(3)+".png", image_temp)

		image_temp = cv2.resize(image_temp, (160, 120))
		cv2.imwrite(path_database+"/frame_160_120_"+str(vec).zfill(3)+".png", image_temp)

		image_temp = cv2.resize(image_temp, (80, 60))
		cv2.imwrite(path_database+"/frame_080_060_"+str(vec).zfill(3)+".png", image_temp)

		image_temp = cv2.resize(image_temp, (40, 30))
		cv2.imwrite(path_database+"/frame_040_030_"+str(vec).zfill(3)+".png", image_temp)

		image_temp = cv2.resize(image_temp, (20, 15))
		cv2.imwrite(path_database+"/frame_020_015_"+str(vec).zfill(3)+".png", image_temp)

def fun_initialize_gray_patches_image_db(cam, path_database, database_size):
	'''
	The function saves the first "database_size" frames for to initialize the 
	image database. 
	Here we initialize the image for the db as grayscales patches.
	'''

	# take one frame
	(grabbed, frame) = cam.read()
	frame = cv2.flip(frame,1)

	# create the random values
	data_lut = np.round(np.linspace(0,255,32))
	
	for vec in np.arange(len(data_lut)):
		image_temp = np.ones(np.shape(frame), dtype=float)
		image_temp[:,:,0] = image_temp[:,:,0] * data_lut[vec] 
		image_temp[:,:,1] = image_temp[:,:,1] * data_lut[vec] 
		image_temp[:,:,2] = image_temp[:,:,2] * data_lut[vec]

		cv2.imwrite(path_database+"/frame_640_480_"+str(vec).zfill(3)+".png", image_temp)

		image_temp = cv2.resize(image_temp, (320, 240)) 
		cv2.imwrite(path_database+"/frame_320_240_"+str(vec).zfill(3)+".png", image_temp)

		image_temp = cv2.resize(image_temp, (160, 120))
		cv2.imwrite(path_database+"/frame_160_120_"+str(vec).zfill(3)+".png", image_temp)

		image_temp = cv2.resize(image_temp, (80, 60))
		cv2.imwrite(path_database+"/frame_080_060_"+str(vec).zfill(3)+".png", image_temp)

		image_temp = cv2.resize(image_temp, (40, 30))
		cv2.imwrite(path_database+"/frame_040_030_"+str(vec).zfill(3)+".png", image_temp)

		image_temp = cv2.resize(image_temp, (20, 15))
		cv2.imwrite(path_database+"/frame_020_015_"+str(vec).zfill(3)+".png", image_temp)


def fun_create_image_database(path_database, number_rows):
	'''The function reads the images save the db folder and compute the metric for each pathToImages
	The db is return as a list [imagePathAndName avR avG avB]
	'''

	frame_for_mosaic_db = []

	if number_rows == 2:
		image_search_pattern = "320_240"
	elif number_rows == 4:
		image_search_pattern = "160_120"
	elif number_rows == 8:
		image_search_pattern = "080_060"
	elif number_rows == 16:
		image_search_pattern = "040_030"
	elif number_rows == 32:
		image_search_pattern = "020_015"
	else:
		print "Houston...."

	t = time.time()
	for imagePath in glob.glob(path_database+"/*"+image_search_pattern+"*.png"):
		# extract our unique image ID (i.e. the filename)
		k = imagePath[imagePath.rfind("/") + 1:]
		#print k
		# load the image, describe it using our RGB histogram
		# descriptor, and update the index
		image = cv2.imread(imagePath)
		features = describe_mean(image)
		frame_for_mosaic_db.append([k, features])

	tt = time.time()
	print "creation of the db in %1.5f" % (tt -t)

	return frame_for_mosaic_db

def fun_create_image_database2(path_database, number_rows, database_size):
	'''The function does the same things as the one above, but in a smarter way.
	Instead of returning one list with everything we are returning:
	- list of image names that are constituing the db
	- numpy array of size 3 x nb_images where each row is the average value R, G and B of 
	each image.
	In:
		- path_database (str): the path to where the images are saved 
		- number_rows (int): information about the final grid image size
		- database_size (int): the number of images for the "image-LUT"

	Out:
		- list_image_name_db (list of str): list of the image name that constitute the image-LUT
		- metric_image_db (array int): array of list_image_name_db size storing the average color
		value of each image
	'''
    
	list_image_name_db = []
	metric_image_db = np.zeros((3,database_size))
	
	if number_rows == 2:
		image_search_pattern = "320_240"
	elif number_rows == 4:
		image_search_pattern = "160_120"
	elif number_rows == 8:
		image_search_pattern = "080_060"
	elif number_rows == 16:
		image_search_pattern = "040_030"
	elif number_rows == 32:
		image_search_pattern = "020_015"
	else:
		print "Houston...."

	i = 0
	# I load the image in original size but will take on the center area for its value
	for imagePath in glob.glob(path_database+"/*"+"640_480"+"*.png"):
		# extract our unique image ID (i.e. the filename)
		k_original = imagePath[imagePath.rfind("/") + 1:]
		k_tile = k_original[0:6]+image_search_pattern+k_original[-8:-4]+".png"
		image = cv2.imread(imagePath)
		shape_image = np.shape(image)
		ptCenter = (shape_image[1]/2,shape_image[0]/2)

		image_roi = image[ptCenter[1] - (int(image_search_pattern[0:3]) / 2): ptCenter[1] + (int(image_search_pattern[0:3]) / 2),
						  ptCenter[0] - (int(image_search_pattern[4:]) / 2): ptCenter[0] + (int(image_search_pattern[4:]) / 2),:]
		#print np.shape(image_roi)
		features = describe_mean(image_roi)
		list_image_name_db.append(k_tile)
		metric_image_db[:,i] = np.transpose(features)
		i = i + 1

	return list_image_name_db, metric_image_db

def fun_create_image_database3(path_database, number_rows, database_size):
	'''The function does the same things as the one aboveen version 2, but in an even smarter way.
	Instead of returning one list with everything we are returning:
	- list of image names that are constituing the db
	- numpy array of size 3 x nb_images where each row is the average value R, G and B of 
	each image.
	In:
		- path_database (str): the path to where the images are saved 
		- number_rows (int): information about the final grid image size
		- database_size (int): the number of images for the "image-LUT"

	Out:
		- list_image_name_db (list of str): list of the image name that constitute the image-LUT
		- metric_image_db (array int): array of list_image_name_db size storing the average color
		value of each image
	'''
    
	list_image_name_db = []
	metric_image_db = np.zeros((3,database_size))
	
	if number_rows == 2:
		image_search_pattern = "320_240"
	elif number_rows == 4:
		image_search_pattern = "160_120"
	elif number_rows == 8:
		image_search_pattern = "080_060"
	elif number_rows == 16:
		image_search_pattern = "040_030"
	elif number_rows == 32:
		image_search_pattern = "020_015"
	else:
		print "Houston...."

	i = 0
	for imagePath in glob.glob(path_database+"/*"+image_search_pattern+"*.png"):
		# extract our unique image ID (i.e. the filename)
		k = imagePath[imagePath.rfind("/") + 1:]
		#print k
		image = cv2.imread(imagePath)
		features = describe_mean(image)
		list_image_name_db.append(k)
		metric_image_db[:,i] = np.transpose(features)
		i = i + 1

	return list_image_name_db, metric_image_db




	return list_image_name_db, metric_image_db
def describe_mean(frame):
	'''The function computes the average color per rgb channel
	of the given frame as input.
	'''
	rgb_mean = np.zeros(3)
	
	rgb_mean[0] = frame[:,:,0].mean()
	rgb_mean[1] = frame[:,:,1].mean()
	rgb_mean[2] = frame[:,:,2].mean()
	
	return rgb_mean.flatten()

def describe_mean_center(frame):
	'''The function computes the average color per rgb channel
	of the given frame as input BUT only for the center of the image.
	The area is approximatively 25 %  the image surface. 
	'''
	rgb_mean = np.zeros(3)
	ss = np.shape(frame)
	print ss
	rgb_mean[0] = frame[int(ss[0]/8):int(3 * ss[0]/8),int(ss[1]/8):int(3 * ss[1]/8),0].mean()
	rgb_mean[1] = frame[int(ss[0]/8):int(3 * ss[0]/8),int(ss[1]/8):int(3 * ss[1]/8),1].mean()
	rgb_mean[2] = frame[int(ss[0]/8):int(3 * ss[0]/8),int(ss[1]/8):int(3 * ss[1]/8),2].mean()
	#rgb_mean[0] = frame[320 - (160 / 2): 320 + (160 / 2), 240 - (120 / 2): 240 + (120 / 2),0].mean()
	#rgb_mean[1] = frame[320 - (160 / 2): 320 + (160 / 2), 240 - (120 / 2): 240 + (120 / 2),1].mean()
	#rgb_mean[2] = frame[320 - (160 / 2): 320 + (160 / 2), 240 - (120 / 2): 240 + (120 / 2),2].mean()
	print rgb_mean
	return rgb_mean.flatten()

def fun_save_image_for_db(frame, path_database, frame_name):
	'''
	The function saves the new grabbed image in order to use it later.
	'''
	# save original 
	cv2.imwrite(path_database+"/"+frame_name+".png", frame)

def fun_construct_image_from_index_DB(frame, pathToImage, list_db, number_rows, metric_db):
	# It' ugly but it's working...

	if number_rows == 2:
		image_search_pattern = "320_240"
	elif number_rows == 4:
		image_search_pattern = "160_120"
	elif number_rows == 8:
		image_search_pattern = "80_60"
	elif number_rows == 16:
		image_search_pattern = "40_30"
	elif number_rows == 32:
		image_search_pattern = "20_15"
	else:
		print "Houston...."

	# create the mosaic function based from the image_search_pattern
	fun_mosaic = getattr(toolboox, 'constructImageFromIndexDB_'+image_search_pattern)

	mirror_mosaic_frame = fun_mosaic(frame, pathToImage, list_db, number_rows, metric_db)

	return mirror_mosaic_frame

def fun_construct_image_from_index_DB2(frame_video, path_to_image, list_db, number_rows, metric_db):
	'''This function takes a frame as input and return the mosaic version of it as output.

	In:
		- frame_video (array image): an image taken by the webcam or loaded from video file
		- path_to_image (str): the path to where the images of image-LUT are store
		- list_db (list str): list of str for each image name constituing the image-LUT
		- number_rows (int): information about the grid size for the mosaic number_rows x number_rows
		- metric_db (int array): array that store the metric for each image fo the image-LUT

	Out:
		- mirror_mosaic_frame (array image): an image of the same size as frame_video but made of 
		tiny images from list_db
	'''

	if number_rows == 2:
		image_search_pattern = "320_240"
	elif number_rows == 4:
		image_search_pattern = "160_120"
	elif number_rows == 8:
		image_search_pattern = "80_60"
	elif number_rows == 16:
		image_search_pattern = "40_30"
	elif number_rows == 32:
		image_search_pattern = "20_15"
	else:
		print "Houston...."

	# scale image	
	scaled_frame_video = cv2.resize(frame_video, (number_rows, number_rows)) 

	# get the image indexes for each cell
	index_image = fun_give_image_indexes_from_DB2(scaled_frame_video, number_rows, metric_db)

	# construc the mosaic image
	mirror_mosaic_frame = frame_video
	vec_v = np.hstack([np.arange(0,np.shape(frame_video)[0], np.shape(frame_video)[0] / number_rows), 
															 np.shape(frame_video)[0]])
	vec_h = np.hstack([np.arange(0,np.shape(frame_video)[1], np.shape(frame_video)[1] / number_rows), 
															 np.shape(frame_video)[1]])

	vec_ind = np.arange(0,number_rows * number_rows)
	c = 0 
	for i in np.arange(len(vec_v)-1):
		for j in np.arange(len(vec_h)-1):
			mirror_mosaic_frame[vec_v[i]:vec_v[i+1], vec_h[j]:vec_h[j+1]] = cv2.imread(path_to_image+'/'+list_db[index_image[vec_ind[c]]])
			c = c + 1

	return mirror_mosaic_frame

def fun_construct_image_from_index_DB3(frame_video, path_to_image, list_db, number_rows, metric_db):
	'''This function takes a frame as input and return the mosaic version of it as output.

	But instead of scaling down it take as reference the center of the frame according to the tile size.

	In:
		- frame_video (array image): an image taken by the webcam or loaded from video file
		- path_to_image (str): the path to where the images of image-LUT are store
		- list_db (list str): list of str for each image name constituing the image-LUT
		- number_rows (int): information about the grid size for the mosaic number_rows x number_rows
		- metric_db (int array): array that store the metric for each image fo the image-LUT

	Out:
		- mirror_mosaic_frame (array image): an image of the same size as frame_video but made of 
		tiny images from list_db
	'''
	
	shape_image = np.shape(frame_video)
	ptCenter = (shape_image[1]/2,shape_image[0]/2)

	if number_rows == 2:
		image_search_pattern = "320_240"
		ss = np.array([320, 240])

	elif number_rows == 4:
		image_search_pattern = "160_120"
		ss = np.array([160, 120])
	elif number_rows == 8:
		image_search_pattern = "80_60"
		ss = np.array([80, 60])
	elif number_rows == 16:
		image_search_pattern = "40_30"
		ss = np.array([40, 30])
	elif number_rows == 32:
		image_search_pattern = "20_15"
		ss = np.array([20, 15])
	else:
		print "Houston...."

	# scale image	
	scaled_frame_video = cv2.resize(frame_video, (number_rows, number_rows)) 

	# get the image indexes for each cell
	index_image = fun_give_image_indexes_from_DB2(scaled_frame_video, number_rows, metric_db)

	# construc the mosaic image
	mirror_mosaic_frame = frame_video
	vec_v = np.hstack([np.arange(0,np.shape(frame_video)[0], np.shape(frame_video)[0] / number_rows), 
															 np.shape(frame_video)[0]])
	vec_h = np.hstack([np.arange(0,np.shape(frame_video)[1], np.shape(frame_video)[1] / number_rows), 
															 np.shape(frame_video)[1]])

	vec_ind = np.arange(0,number_rows * number_rows)
	c = 0 
	for i in np.arange(len(vec_v)-1):
		for j in np.arange(len(vec_h)-1):
			mirror_mosaic_frame[vec_v[i]:vec_v[i+1], vec_h[j]:vec_h[j+1]] = cv2.imread(path_to_image+'/'+list_db[index_image[vec_ind[c]]])
			c = c + 1

	return mirror_mosaic_frame

def giveImagesFromDB(frame, pathToImage, list_db, number_row):
	'''The function takes a frame as input and gives the mosaic version as output.

	It scales down the frame as pixel version where each pixel is compare to the mosaic 
	images in the list_db.

	It returns an index toward the images to be used for the mosaic
	'''


	# scale the image down
	#t0 = time.time()
	frame_scaled = cv2.resize(frame, (number_row, number_row)) 

	imR = frame_scaled[:,:,0].flatten()
	imG = frame_scaled[:,:,1].flatten()
	imB = frame_scaled[:,:,2].flatten()
	
	##d = np.zeros((np.size(imR),len(list_db)), dtype = imR.dtype)
	
	##for i in np.arange(len(imR)):
	##    for j in np.arange(len(list_db)):
	##       pix = [imR[i], imG[i], imB[i]]
	##        d[i,j] = np.sqrt(np.sum((pix - list_db[j][1])**2))
	
	##index_for_images = np.argmin(d,axis=1)

	nb_block = number_row * number_row
	db_size = len(list_db)
	#imR = np.random.randint(255, size=(1, nb_block))
	#imG = np.random.randint(255, size=(1, nb_block))
	#imB = np.random.randint(255, size=(1, nb_block))
	im_block = np.vstack([imR, imG, imB])

	# Here I'm recasting the db as a matrix of vector, one vector per db image
	#db  = np.random.randint(255, size=(3,db_size))
	db = np.zeros((3,db_size))
	for i in np.arange(len(list_db)):
		db[:,i] = np.transpose(list_db[i][1])

	res = np.zeros(imR.size)

	for i in np.arange(imR.size):
	    single_block = np.transpose(np.tile(im_block[:,i],(db_size,1)))
	    diff = np.sqrt(np.sum(single_block - db,0)**2)
	    res[i] = np.argmin(diff)
	
	#t2 = time.time()
	#print "getting the index in %1.5f" % (t2 - t0)

	index_for_images = np.uint8(res)
	#print res.dtype
	return index_for_images

def fun_give_image_indexes_from_DB2(frame_scaled, number_row, metric_db):
	'''The function takes the scales frame as input and return the index for each cell.

	In:
		- frame_scaled (array image): scaled version of the webcam frame
		- number_rows (int): information about the grid size for the mosaic number_rows x number_rows
		- metric_db (int array): array that store the metric for each image fo the image-LUT

	Out:
		- index_for_images (array): array of index to build the mosaic frame
		tiny images from list_db
	'''

	imR = frame_scaled[:,:,0].flatten()
	imG = frame_scaled[:,:,1].flatten()
	imB = frame_scaled[:,:,2].flatten()
	
	nb_block = number_row * number_row
	im_block = np.vstack([imR, imG, imB])

	res = np.zeros(imR.size)

	for i in np.arange(imR.size):
	    single_block = np.transpose(np.tile(im_block[:,i],(np.shape(metric_db)[1],1)))
	    diff = np.sqrt(np.sum(single_block - metric_db,0)**2)
	    res[i] = np.argmin(diff)
	 
	index_for_images = np.uint8(res)
	return index_for_images

def constructImageFromIndexDB(frame, pathToImage, list_db, number_row):
	'''In the previous function we computed the images corresponding to the average color area.

	Now we can replace the tile by the corresponding images'''

	index_image = giveImagesFromDB(frame, pathToImage, list_db, number_row)

	mirror_mosaic_frame = frame

	mirror_mosaic_frame[0:240,0:320,:] = cv2.imread('frame_db/'+list_db[index_image[0]][0])
	mirror_mosaic_frame[240:,0:320,:] = cv2.imread('frame_db/'+list_db[index_image[1]][0])
	mirror_mosaic_frame[0:240,320:,:] = cv2.imread('frame_db/'+list_db[index_image[2]][0])
	mirror_mosaic_frame[240:,320:,:] = cv2.imread('frame_db/'+list_db[index_image[3]][0])

	return mirror_mosaic_frame

def fun_update_image_db(frame_video, path_to_image, list_db, metric_db, number_rows):
	'''The function update the database information by adding  a new image to the existing
	databse and removing one element of the db such that db keeps the same size.
	In:
		- frame (array): the latest images recorded
		- path_to_image: the path were the images are stored
		- list_db (list): list of str with the image name constituing the db 
		- metric_db (array): array of metric for each image of the db

	Out:
		- list_db (list): same as input
		- metric_db (array): same as input

	'''

	# get metric from new image
	features = np.transpose(describe_mean(frame_video))

 	# add element to db
 	new_metric_db = np.zeros(np.shape(metric_db))
 	new_metric_db[:,0:np.shape(new_metric_db)[1]-1] = metric_db[:,1:]
 	#print new_metric_db, np.shape(new_metric_db)
 	
	# replace image in the db
	new_frame_for_db = cv2.imread(path_to_image+"new_frame.png")	

	if number_rows == 2:
		new_frame_resized_for_db = cv2.resize(new_frame_for_db, (320, 240))
		image_search_pattern = "320_240"
	elif number_rows == 4:
		new_frame_resized_for_db = cv2.resize(new_frame_for_db, (160, 120))
		image_search_pattern = "160_120"
	elif number_rows == 8:
		new_frame_resized_for_db = cv2.resize(new_frame_for_db, (80, 60))
		image_search_pattern = "080_060"
	elif number_rows == 16:
		new_frame_resized_for_db = cv2.resize(new_frame_for_db, (40, 30))
		image_search_pattern = "040_030"
	elif number_rows == 32:
		new_frame_resized_for_db = cv2.resize(new_frame_for_db, (20, 15))
		image_search_pattern = "020_015"	
	else:
		print "Houston...."
	# and we save this image as the last one of
	counter = int(list_db[-1][-7:-4])
	new_name = "frame_"+image_search_pattern+"_"+str(counter+1).zfill(3)+".png"
	list_db.append(new_name)
	# remove the first image of the database
	os.remove(path_to_image+list_db[0])
	list_db.pop(0)
	cv2.imwrite(path_to_image+list_db[-1],new_frame_resized_for_db)

	return list_db, new_metric_db

