# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import cv2
import dlib
import os


def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear


def mouth_aspect_ratio(mouth):

	A=dist.euclidean(mouth[1],mouth[11])
	B=dist.euclidean(mouth[2],mouth[10])
	C=dist.euclidean(mouth[3],mouth[9])
	D=dist.euclidean(mouth[4],mouth[8])
	E=dist.euclidean(mouth[5],mouth[7])

	F=dist.euclidean(mouth[0],mouth[6])

	mar=(A+B+C+D+E)/(5.0*F)
	
	return mar


# construct the argument parse and parse the arguments

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 40

MOUTH_AR_THRESH=0.50
MOUTH_AR_CONSEC_FRAMES = 15

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
COUNTER_MOUTH=0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mouthStart,mouthEnd)=face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(0).start()
time.sleep(1.0)


# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	now=time.time()
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	if gray.all()==False:
		pass
        #t = Thread(target=cover_camera,args=(now,))
		#t.deamon = True
		#t.start()


	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mouth=shape[mouthStart:mouthEnd]

		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		
		mar=mouth_aspect_ratio(mouth)


		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull =cv2.convexHull(mouth)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if mar>MOUTH_AR_THRESH:
			COUNTER_MOUTH +=1
			if COUNTER_MOUTH>=MOUTH_AR_CONSEC_FRAMES:
				if not ALARM_ON:
					ALARM_ON=True
					if ALARM_ON:
						t1=Thread(target=sound_alarm,args=("alarm.wav",))
						t1.deamon=True
						t1.start()
					
					cv2.putText(frame, "YAWN ALERT!", (10, 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			COUNTER_MOUTH=0
			ALARM_ON=False

		if ear < EYE_AR_THRESH:
			COUNTER += 1

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					if ALARM_ON:
						t = Thread(target=sound_alarm,
							args=("drowsiness.mp3",))
						t.deamon = True
						t.start()

				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			ALARM_ON = False

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 50),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(10) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import cv2
import dlib

def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear


def mouth_aspect_ratio(mouth):

	A=dist.euclidean(mouth[1],mouth[11])
	B=dist.euclidean(mouth[2],mouth[10])
	C=dist.euclidean(mouth[3],mouth[9])
	D=dist.euclidean(mouth[4],mouth[8])
	E=dist.euclidean(mouth[5],mouth[7])

	F=dist.euclidean(mouth[0],mouth[6])

	mar=(A+B+C+D+E)/(5.0*F)
	
	return mar

def cover_camera(color):
	pass


 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.25#orj 0.3
EYE_AR_CONSEC_FRAMES = 40

MOUTH_AR_THRESH=0.50#will change
MOUTH_AR_CONSEC_FRAMES = 15

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
COUNTER_MOUTH=0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mouthStart,mouthEnd)=face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(0).start()
time.sleep(1.0)


# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mouth=shape[mouthStart:mouthEnd]

		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		
		mar=mouth_aspect_ratio(mouth)


		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull =cv2.convexHull(mouth)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if mar>MOUTH_AR_THRESH:
			COUNTER_MOUTH +=1
			if COUNTER_MOUTH>=MOUTH_AR_CONSEC_FRAMES:
				if not ALARM_ON:
					ALARM_ON=True
					if ALARM_ON:
						t1=Thread(target=sound_alarm,args=("alarm.wav",))
						t1.deamon=True
						t1.start()
					
					cv2.putText(frame, "YAWN ALERT!", (10, 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			COUNTER_MOUTH=0
			ALARM_ON=False

		if ear < EYE_AR_THRESH:
			COUNTER += 1

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					if ALARM_ON:
						t = Thread(target=sound_alarm,
							args=("drowsiness.mp3",))
						t.deamon = True
						t.start()

				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			ALARM_ON = False

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 50),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(10) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()