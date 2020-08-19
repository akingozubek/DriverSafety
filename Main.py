# import the necessary packages
import os
import time
from threading import Thread

import cv2
import dlib
import imutils
import numpy as np
import playsound
from gtts import gTTS
from imutils import face_utils
from imutils.video import VideoStream
from scipy.spatial import distance as dist

#Sesler->Sounds Klasöründe
#Modeller->Models Klasöründe
#Resimler->Images Klasörüne

class DriverSafety():

    def __init__(self,camera=0):

        #Eyes aspect ratio thresholds and frame count
        self.EYE_AR_THRESH = 0.25#+- changeable
        self.EYE_AR_CONSEC_FRAMES = 15#+- changeable

        #Counters
        self.COUNTER=0
        self.COVER_COUNTER=0
        self.ATTENTION_COUNTER=0
        self.SMOKE_COUNTER=0
        self.PHONE_COUNTER=0


        #camera and text font
        self.camera=cv2.VideoCapture(camera)
        self.font=cv2.FONT_HERSHEY_PLAIN

        #log and image save adaptibility
        self.last_err=""
        self.last_err_time=0

        #Files Paths
        #self.alert_path="Sounds/"
        #self.save_image_path="Images/"
        #self.models_path="Models/"
        
        #Create some directory
        self.alert_path=self.createPath("Sounds/")
        self.save_image_path=self.createPath("Images/")
        self.models_path=self.createPath("Models/")
        
        #yolo models-facial ladmarks models
        self.models()

        #start camera
        self.startVideoStream(self.camera)
        #self.startThreads(self.startVideoStream,args_=(self.camera,))


    #create directory if is not exist.
    def createPath(self,path):

        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            return path
            

    #Yolo Models/Facial Landmarks
    def models(self):
        
        #dlib model
        self.face_landmarks=self.models_path+"shape_predictor_68_face_landmarks.dat"
        self.detector=dlib.get_frontal_face_detector()
        self.predictor=dlib.shape_predictor(self.face_landmarks)
        
        #eyes location index
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        #yolo model
        #self.net=cv2.dnn.readNet(self.models_path+"yolov3-tiny_training_last.weights",self.models_path+"yolov3-tiny_testing.cfg")
        self.net=cv2.dnn.readNet(self.models_path+"yolov3_training_last.weights",self.models_path+"yolov3_testing.cfg")

        #classes
        with open(self.models_path+"classes.names","r") as f:
            self.classes=f.read().splitlines()


    #threads start function
    def startThreads(self,_target,_args=()):

        t=Thread(target=_target,args=_args)
        t.daemon=True
        t.start()
        t.join()


    #Camera Run
    def startVideoStream(self,camera):
        
        time.sleep(2.0)#waiting for camera build up

        self.logFile("Camera Opened")#Camera Open Log
        
        while True:

            ret,self.frame=camera.read()#read camera 
            
            #if camera does not respond, shuts down system
            if not ret:
                break

            self._time=time.time()#read time every frame

            #resize frame
            self.frame = imutils.resize(self.frame, width=480,height=480)
            #grayscale frame
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            
            #if camera is blocked ->changeable
            if not self.gray.any():
                self.startThreads(self.controlCameraBlocked)

            if np.mean(self.gray)/255 < 0.5:
                self.gray=cv2.equalizeHist(self.gray)

            
            #start object detection control, facial landmarks control and driver attention detection
            self.startThreads(self.objectDetection)
            self.startThreads(self.faceAndEyesDetection)

            self.startThreads(self.attentionDetection)
            self.startThreads(self.phoneDetection)
            self.startThreads(self.smokeDetection)

            #show camera
            cv2.imshow("Camera",self.gray)
            
            #press ESC or Q close camera
            self.key=cv2.waitKey(1) & 0xff
            if self.key==27 or self.key==ord("q"):
                break

        #stop processing
        self.stopVideoStream()


    def controlCameraBlocked(self):

        #if camera blocked, when reach specified time, run warning and save image. 
        self.COVER_COUNTER+=1
        if self.COVER_COUNTER>5:
            time.sleep(1.0)
            self.saveImage(self.frame,"Camera Blocked")
            self.warning("BlockedCameraWarning.mp3")
            #time.sleep(5.0)
        if self.gray.any():
            self.COVER_COUNTER=0


    #Yolo Object Detection
    def objectDetection(self):
        
        height,width=self.frame.shape[0],self.frame.shape[1]
        
        #will be drawn box list, scores list and object id list
        boxes=[]
        confidences=[]
        class_ids=[]

        #image to blob and detect object
        blob=cv2.dnn.blobFromImage(self.frame,1/255,(416,416),(0,0,0),swapRB=True,crop=False)
        self.net.setInput(blob)
        out_layers_name=self.net.getUnconnectedOutLayersNames()
        layer_outs=self.net.forward(out_layers_name)


        #if there are any object
        for out in layer_outs:
            for detection in out:
                score=detection[5:]
                class_id=np.argmax(score)#object index
                confidence=score[class_id]#score is detected object
                
                #if score %50 coordination and boxes process
                if confidence>0.5:
                    center_x=int(detection[0]*width)
                    center_y=int(detection[0]*height)
                    
                    w=int(detection[2]*width)
                    h=int(detection[2]*height)
                    
                    x=int(center_x-w/2)
                    y=int(center_y-h/2)

                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        self.control_class_id=class_ids.copy()

        idx=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
        colors=np.random.uniform(0,255,size=(len(boxes),3))
    
        #show boxes and text
        #try:
        try:
            for i in idx.flatten():
                x,y,w,h=boxes[i]
                label=str(self.classes[class_ids[i]])
                print(label)
                confidence=str(round(confidences[i],2))
                color=colors[i]
                cv2.rectangle(self.gray,(x,y),(x+w,y+h),color,1)
                cv2.putText(self.gray,label+confidence,(x,y+20),self.font,2,(255,255,255),2)
            #self.putTextVideoStream(label,confidence,x,y+10)
        except:
            pass


    #Calculate eyee aspect ratio
    def findEyeAspectRatio(self,eye):

        first_height = dist.euclidean(eye[1], eye[5])
        second_height = dist.euclidean(eye[2], eye[4])
        eye_width = dist.euclidean(eye[0], eye[3])

        self.eye_aspect_ratio = (first_height + second_height) / (2.0 * eye_width)

        return self.eye_aspect_ratio

    #Face and Eye detection with dlib
    def faceAndEyesDetection(self):

        self.rects=self.detector(self.gray,0)

        for rect in self.rects:

            shape = self.predictor(self.gray, rect)
            shape = face_utils.shape_to_np(shape)


            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]

            leftEAR = self.findEyeAspectRatio(leftEye)
            rightEAR = self.findEyeAspectRatio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            self.drowsinessDetection(ear)
            self.putTextVideoStream("EAR",ear,250,30)


    #if driver look another direction long time, run warning and save image
    def attentionDetection(self):


        try:
            control=True if 0 in self.control_class_id else False

            if not self.rects and control:
                self.ATTENTION_COUNTER+=1
                if self.ATTENTION_COUNTER>10:
                    self.saveImage(self.frame,"Attention")
                    self.warning("AttentionWarning.mp3")
                    #time.sleep(5.0)
                    
            else:
                self.ATTENTION_COUNTER=0
        except:
            pass


    #if detect cigarette, run warning and save image
    def smokeDetection(self):

        try:
            control=True if 2 in self.control_class_id else False
            if control:
                self.SMOKE_COUNTER+=1
                if self.SMOKE_COUNTER>5:
                    self.saveImage(self.frame,"Smoking")
                    self.warning("SmokeWarning.mp3")
                    #time.sleep(5.0)
            else:
                self.SMOKE_COUNTER=0
        except:
            pass


    #if detect phone, run warning and save image    
    def phoneDetection(self):

        try:
            control=True if 1 in self.control_class_id else False
            if control:
                self.PHONE_COUNTER+=1
                if self.PHONE_COUNTER>5:
                    self.saveImage(self.frame,"Phone")
                    self.warning("PhoneWarning.mp3")
                    #time.sleep(5.0)
            else:
                self.PHONE_COUNTER=0
        except:
            pass


    #if eyes aspect ratio < identified threshold. run warning and save image.
    def drowsinessDetection(self,ear):

        if ear < self.EYE_AR_THRESH:
            self.COUNTER += 1

            if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                self.saveImage(self.frame,"Drowsiness")
                self.warning("Drowsiness.mp3")
                #time.sleep(3.0)
        else:
            self.COUNTER = 0

    #play warning sounds
    def warning(self,file):
        path=self.alert_path+file
        playsound.playsound(path)
        time.sleep(2.0)


    #if detected any anomaly, save it.
    def saveImage(self,frame,err):
        if err==self.last_err:
            if self._time-self.last_err_time>5:
                img="{}/{}_{}.jpg".format(self.save_image_path,err,time.time())
                cv2.imwrite(img,frame)
                self.sendImage(image,True)
        else:
            img="{}/{}_{}.jpg".format(self.save_image_path,err,time.time())
            cv2.imwrite(img,frame)
            self.sendImage(image,True)
        
        self.logFile(err)
        self.last_err=err
        self.last_err_time=time.time()


    def logFile(self,err):

        if err==self.last_err:
            if self._time-self.last_err_time>5:
                date=time.strftime("%x")
                _time=time.strftime("%X")
                with open("log.txt","a") as f:
                    current_log=date+" "+_time+" "+err+"\n"
                    f.write(current_log)
        else:
            date=time.strftime("%x")
            _time=time.strftime("%X")
            with open("log.txt","a") as f:
                current_log=date+" "+_time+" "+err+"\n"
                f.write(current_log)
        

    #put text camera screen, may be deleted
    def putTextVideoStream(self,text,value,x,y):
        cv2.putText(self.gray, text+ " : {:.3f}".format(value), (x, y),
        self.font, 2, (0, 0, 0), 2)
   

    #release camera, close camera window and log it.
    def stopVideoStream(self):

        self.logFile("Camera Closed")
        self.camera.release()
        cv2.destroyAllWindows()





if __name__=="__main__":
    driver=DriverSafety()
