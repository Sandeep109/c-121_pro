import cv2
import numpy as np
import time

# replace the red pixels ( or undesired area ) with
# background pixels to generate the invisibility feature.

## 1. Hue: This channel encodes color information. Hue can be
# thought of an angle where 0 degree corresponds to the red color,
# 120 degrees corresponds to the green color, and 240 degrees
# corresponds to the blue color.

## 2. Saturation: This channel encodes the intensity/purity of color.
# For example, pink is less saturated than red.

## 3. Value: This channel encodes the brightness of color.
# Shading and gloss components of an image appear in this
# channel reading the videocapture video

# in order to check the cv2 version

# taking video.mp4 as input.
# Make your path according to your needs
capture_video = cv2.VideoCapture(0)
count = 0
# capturing the background in range of 60
# you should have video that have some seconds
# dedicated to background frame so that it
# could easily save the background image

path = r'G:\PY FIles\c-121\bangkok.jpg'
image = cv2.imread(path) # flipping of the frame

# we are reading from video
while (capture_video.isOpened()):
	return_val, frame = capture_video.read()
	if not return_val :
		break
	count = count + 1

	#-------------------------------------BLOCK----------------------------#
	# ranges should be carefully chosen
	# setting the lower and upper range for mask1
	u_black = np.array([104, 153, 70])	
	l_black = np.array([30, 30, 0])

	frame = cv2.resize(frame, (680,480))
	image = cv2.resize(image, (680,480))
	
	mask = cv2.inRange(frame, l_black, u_black)
	res = cv2.bitwise_and(frame, frame, mask = mask)

	f = frame - res
	f = np.where(f == 0, image, f)
	#----------------------------------------------------------------------#

	cv2.imshow("INVISIBLE MAN", image)
	cv2.imshow("INVISIBLE MAN", f)

	k = cv2.waitKey(10)
	if k == 27:
		break

	# the above block of code could be replaced with
	# some other code depending upon the color of your cloth
	#mask1 = mask1 + mask2

	# Refining the mask corresponding to the detected red color
	#mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3),
	#									np.uint8))
	#mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8))
	#mask2 = cv2.bitwise_not(mask1)

	# Generating the final output
	#res1 = cv2.inRange(background, background, mask = mask1)
	#res2 = cv2.bitwise_and(img, img, mask = mask2)

capture_video.release()

cv2.destroyAllWindows()