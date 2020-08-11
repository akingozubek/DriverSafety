# import the necessary packages
from scipy.spatial import distance as dist
from threading import Thread
from gtts import gTTS
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import numpy as np
import playsound
import time
import cv2
import dlib
import os


#Sesler->Sounds Klasöründe
#Modeller->Models Klasöründe
#Resimler->Images Klasörüne

class DriverSafety():

    def __init__(self,camera=0):
        
        self.EYE_AR_THRESH = 0.25#+- changeable
        self.EYE_AR_CONSEC_FRAMES = 25#+- changeable

        #Counters
        self.COUNTER=0
        self.COVER_COUNTER=0
        self.ATTENTION_COUNTER=0
        self.SMOKE_COUNTER=0
        self.PHONE_COUNTER=0

        #camera
        self.camera=cv2.VideoCapture(camera)

        #log and image save adaptibility
        self.last_err=""
        self.last_err_time=0
        self.log_time=0

        #Files Paths
        #self.alert_path="Sounds/"
        #self.save_image_path="Images/"
        #self.models_path="Models/"
        
        #Create some dictionary
        self.alert_path=self.createPath("Sounds/")
        self.save_image_path=self.createPath("Images/")
        self.models_path=self.createPath("Models/")
        
        #yolo models-facial ladmarks models
        self.models()
        self.facialLandmarksIDX()
        self.readObjectClassNames()

        #start camera
        self.startThreads(self.startVideoStream,args_=(self.camera,))

    #create directory if is not exist.
    def createPath(self,path):
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            return path
            
    #threads start function
    def startThreads(self,target_,args_=()):
        t=Thread(target=target_,args=args_)
        t.daemon=True
        t.start()
        t.join()

    #play warning sounds
    def warning(self,file):
        path=self.alert_path+file
        playsound.playsound(path)
    #Yolo Models/Facial Landmarks
    def models(self):
        self.face_landmarks=self.models_path+"shape_predictor_68_face_landmarks.dat"
        self.detector=dlib.get_frontal_face_detector()
        self.predictor=dlib.shape_predictor(self.face_landmarks)
        self.net=cv2.dnn.readNet(self.models_path+"yolov3-tiny.weights",self.models_path+"yolov3-tiny.cfg")


    def facialLandmarksIDX(self):
        landmarks_3d_list = [
            np.array([
                [ 0.000,  0.000,   0.000],    # Nose tip
                [ 0.000, -8.250,  -1.625],    # Chin
                [-5.625,  4.250,  -3.375],    # Left eye left corner
                [ 5.625,  4.250,  -3.375],    # Right eye right corner
                [-3.750, -3.750,  -3.125],    # Left Mouth corner
                [ 3.750, -3.750,  -3.125]     # Right mouth corner 
            ], dtype=np.double),
            np.array([
                [ 0.000000,  0.000000,  6.763430],   # 52 nose bottom edge
                [ 6.825897,  6.760612,  4.402142],   # 33 left brow left corner
                [ 1.330353,  7.122144,  6.903745],   # 29 left brow right corner
                [-1.330353,  7.122144,  6.903745],   # 34 right brow left corner
                [-6.825897,  6.760612,  4.402142],   # 38 right brow right corner
                [ 5.311432,  5.485328,  3.987654],   # 13 left eye left corner
                [ 1.789930,  5.393625,  4.413414],   # 17 left eye right corner
                [-1.789930,  5.393625,  4.413414],   # 25 right eye left corner
                [-5.311432,  5.485328,  3.987654],   # 21 right eye right corner
                [ 2.005628,  1.409845,  6.165652],   # 55 nose left corner
                [-2.005628,  1.409845,  6.165652],   # 49 nose right corner
                [ 2.774015, -2.080775,  5.048531],   # 43 mouth left corner
                [-2.774015, -2.080775,  5.048531],   # 39 mouth right corner
                [ 0.000000, -3.116408,  6.097667],   # 45 mouth central bottom corner
                [ 0.000000, -7.415691,  4.070434]    # 6 chin corner
            ], dtype=np.double),
            np.array([
                [ 0.000000,  0.000000,  6.763430],   # 52 nose bottom edge
                [ 5.311432,  5.485328,  3.987654],   # 13 left eye left corner
                [ 1.789930,  5.393625,  4.413414],   # 17 left eye right corner
                [-1.789930,  5.393625,  4.413414],   # 25 right eye left corner
                [-5.311432,  5.485328,  3.987654]    # 21 right eye right corner
            ], dtype=np.double)
        ]
    
    # 2d facial landmark list
        lm_2d_index_list = [
            [30, 8, 36, 45, 48, 54],
            [33, 17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8], # 14 points
            [33, 36, 39, 42, 45] # 5 points
        ]
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        

    #Camera Run
    def startVideoStream(self,camera):
        
        time.sleep(2.0)#waiting for camera build up
        self.logFile("Camera Opened")#Camera Open Log
        

        while True:
            ret,self.frame=camera.read()#read camera 
            
            #if camera no respond, close system
            if not ret:
                break

            self._time=time.time()#read time every frame

            #resize frame
            self.frame = imutils.resize(self.frame, width=480,height=480)
            #grayscale frame
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            #self.frame=np.dstack([self.gray,self.gray,self.gray])
            
            #if camera is blocked ->changeable
            if not self.gray.any():
                self.startThreads(self.controlCameraBlocked)
 
            
            #start object detection control, facial landmarks control and driver attention detection
            self.startThreads(self.objectDetection)
            self.startThreads(self.faceAndEyesDetection)
            self.startThreads(self.attentionDetection)

            #show camera
            cv2.imshow("Camera",self.frame)
            
            #press ESC or Q close camera
            self.key=cv2.waitKey(1) & 0xff
            if self.key==27 or self.key==ord("q"):
                break
        #stop processing
        self.stopVideoStream()
    
    #put text camera screen, may be deleted
    def putTextVideoStream(self,text,value,x,y):
        font=cv2.FONT_HERSHEY_PLAIN
        cv2.putText(self.frame, text+ " : {:.3f}".format(value), (x, y),
        font, 2, (0, 0, 0), 2)
    
    
    #if camera blocked, when reach specified time, run warning and save image. 
    def controlCameraBlocked(self):
        self.COVER_COUNTER+=1
        if self.COVER_COUNTER>5:
            time.sleep(1.0)
            self.warning("BlockedCameraWarning.mp3")
            self.saveImage(self.frame,"Camera Blocked")
            #time.sleep(5.0)
        if self.gray.any():
            self.COVER_COUNTER=0
            
    #will be detected objects.
    def readObjectClassNames(self):
        with open(self.models_path+"coco.names","r") as f:
            self.classes=f.read().splitlines()
        

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
                self.control_class_id=class_id.copy()#objects control variable
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


        idx=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
        colors=np.random.uniform(0,255,size=(len(boxes),3))
       
        #show boxes and text
        try:
            for i in idx.flatten():
                x,y,w,h=boxes[i]
                label=str(self.classes[class_ids[i]])
                confidence=str(round(confidences[i],2))
                color=colors[i]
                cv2.rectangle(self.frame,(x,y),(x+w,y+h),color,1)
                cv2.putText(self.frame,label+" : "+confidence,(x,y+20),font,2,(255,255,255),2)
                #self.putTextVideoStream(label,confidence,x,y+10)
        except:
            pass

    #if driver look another direction long time, run warning and save image
    def attentionDetection(self):
        try:
            if not self.rects and self.control_class_id==0:
                self.ATTENTION_COUNTER+=1
                if self.ATTENTION_COUNTER>30:
                    self.warning("AttentionWarning.mp3")
                    self.saveImage(self.frame,"Attention")
                    #time.sleep(5.0)
            else:
                self.ATTENTION_COUNTER=0
        except:
            pass

    #if detect cigarette, run warning and save image
    def smokeDetection(self):
        try:
            if self.control_class_id==3:
                self.SMOKE_COUNTER+=1
                if self.SMOKE_COUNTER>10:
                    self.warning("SmokeWarning")
                    self.saveImage(self.frame,"Smoke")
                    #time.sleep(5.0)
            else:
                self.SMOKE_COUNTER=0
        except:
            pass

    #if detect phone, run warning and save image    
    def phoneDetection(self):
        try:
            if self.control_class_id==2:
                self.PHONE_COUNTER+=1
                if self.PHONE_COUNTER>10:
                    self.warning("PhoneWarning.mp3")
                    self.saveImage(self.frame,"Phone")
                    #time.sleep(5.0)
            else:
                self.PHONE_COUNTER=0
        except:
            pass
    
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

    #Calculate eyee aspect ratio
    def findEyeAspectRatio(self,eye):

        first_height = dist.euclidean(eye[1], eye[5])
        second_height = dist.euclidean(eye[2], eye[4])
        eye_width = dist.euclidean(eye[0], eye[3])

        self.eye_aspect_ratio = (first_height + second_height) / (2.0 * eye_width)

        return self.eye_aspect_ratio

    #if eyes aspect ratio < identified threshold. run warning and save image.
    def drowsinessDetection(self,ear):
 
        if ear < self.EYE_AR_THRESH:
            self.COUNTER += 1

            if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                self.startThreads(target_=self.saveImage,args_=(self.frame,"drowsiness",))
                self.startThreads(target_=self.warning,args_=("drowsiness.mp3",))

                #time.sleep(3.0)
        else:
            self.COUNTER = 0

    #time calculation will be added.
    def timeCounter(self):
        pass

    #if detected any anomaly, save it.
    def saveImage(self,frame,err):
        if err==self.last_err:
            if self._time-self.last_err_time>5:
                cv2.imwrite(self.save_image_path+err+".jpg",frame)
        else:
            cv2.imwrite(self.save_image_path+err+".jpg",frame)
        
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
        
        
   #aynıysa ->5~10 saniye sonra log tut.
   #değilse ->hemen log tut.


   #create Sounds File -> may be deleted.
    def createSound(self,text,filename):
        voice=gTTS(text,lang="en")
        voice.save("Sounds/"+filename)

    #release camera, close camera window and log it.
    def stopVideoStream(self):

        self.logFile("Camera Closed")
        self.camera.release()
        cv2.destroyAllWindows()





if __name__=="__main__":
    driver=DriverSafety()
