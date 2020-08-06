# import the necessary packages
from scipy.spatial import distance as dist
from threading import Thread
from gtts import gTTS
from imutils.video import VideoStream,FPS
from imutils import face_utils
import imutils
import numpy as np
import playsound
import time
import cv2
import dlib
import sys


#Sesler->Sounds Klasöründe
#Modeller->Models Klasöründe
#Resimler->Images Klasörüne

class DriverSafety():
    def __init__(self,face_landmarks,queueSize=128,camera=0):
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 40
        self.COUNTER=0
        self.COVER_COUNTER=0
        self.face_landmarks=face_landmarks
        self.camera=cv2.VideoStream(camera)

        self.alert_path="Sounds/"
        self.save_image_path="Images/"
        self.models_path="Models/"

    def warning(self,file):
        playsound.playsound(self.alert_path+file)

    def models(self,face_landmarks):
        self.face_landmarks=self.models_path+"shape_predictor_68_face_landmarks.dat"
        self.detector=dlib.get_frontal_face_detector()
        self.predictor=dlib.shape_predictor(face_landmarks)
    
    def startVideoStream(self,camera):
        while True:
            ret,frame=cam.read()
        	if not ret:
		        break
        	frame = imutils.resize(frame, width=450)
        	height,width,_=frame.shape
        	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not gray.any():
                controlCameraBlocked(gray)
            
            cv2.imshow("Camera",frame)
            key=cv2.waitKey(1) &0xff
            if key==27 or key==ord("q"):
                break
        stopVideoStream()
    def putTextVideoStream(self,frame,ear):
        cv2.putText(frame, "EYES-AR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def controlCameraBlocked(self,gray):
        self.COVER_CAMERA+=1
        if COVER_COUNTER>5:
            #t1 = Thread(target=warning_alert,args=("Sounds/BlockedCameraWarning.mp3",))
			#t1.daemon = True
			#t1.start()
			#time.sleep(3.0)
            warning("BlockedCameraWarning.mp3")
        else:
            COVER_COUNTER=0
    def objectDetection(self,):

    
        
    def faceAndEyesDetection(self,rects):
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.rects=self.detector(gray,0)
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

		    leftEAR = eye_aspect_ratio(leftEye)
		    rightEAR = eye_aspect_ratio(rightEye)
		# average the eye aspect ratio together for both eyes
		    ear = (leftEAR + rightEAR) / 2.0
		


		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		    leftEyeHull = cv2.convexHull(leftEye)
		    rightEyeHull = cv2.convexHull(rightEye)
    
    def drowsinessDetection(self,gray):
 
        if ear < EYE_AR_THRESH:
			COUNTER += 1

			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				cv2.imwrite("frame.jpg",frame)
				t2 = Thread(target=warning_alert,
					args=("Sounds/drowsiness.mp3",))
				t2.daemon = True
				t2.start()
				time.sleep(3.0)

				# draw an alarm on the frame

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0


    def findEyeAspectRatio(self,eye):

    	# compute the euclidean distances between the two sets of
	    # vertical eye landmarks (x, y)-coordinates
	    first_height = dist.euclidean(eye[1], eye[5])
	    second_height = dist.euclidean(eye[2], eye[4])

    	# compute the euclidean distance between the horizontal
    	# eye landmark (x, y)-coordinates
    	eye_width = dist.euclidean(eye[0], eye[3])

    	# compute the eye aspect ratio
    	eye_aspect_ratio = (first_height + second_height) / (2.0 * eye_width)

    	# return the eye aspect ratio
    	return eye_aspect_ratio

    def drawHull(self,):
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    def timeCounter(self,):
    
    def saveImage(self,frame,err):
        cv2.imwrite(self.save_image_path+err+".jpg",frame)

    def createSound(self,text,filename):
        voice=gTTS(text,lang="en")

        voice.save("Sounds/"+filename)

    def stopVideoStream(self,):
        cam.release()
        cv2.destroyAllWindows()