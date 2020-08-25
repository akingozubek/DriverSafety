# import the necessary packages
import os
import time
from threading import Thread

import cv2
import dlib
import imutils
import numpy as np
import playsound
from imutils import face_utils
from imutils.video import VideoStream
import base64
import json
from scipy.spatial import distance as dist

#Sesler->Sounds Klasöründe
#Modeller->Models Klasöründe
#Resimler->Images Klasörüne

class DriverSafety():

    def __init__(self,camera=0):
        """ 
        Document will be added.
        """
        #Eyes aspect ratio thresholds and frame count
        self.eyes_ar_threshold = 0.25#+- changeable
        self.eye_ar_consec_frames = 5#+- changeable

        #counters
        self.drowsiness_counter=0
        self.cover_counter=0
        self.attention_counter=0
        self.smoke_counter=0
        self.phone_counter=0


        #camera and text font
        self.camera=cv2.VideoCapture(camera)
        self.font=cv2.FONT_HERSHEY_PLAIN

        #for saving all anomalies run time.
        self.anomalies=dict()

        #log and image save adaptibility
        self.last_err=""
        self.last_err_time=0
        
        #Create some directory
        self.alert_path=self.createPath("Sounds/")
        self.save_image_path=self.createPath("Images/")
        self.models_path=self.createPath("Models/")
        
        #yolo models-facial ladmarks models
        self.models()

        #start camera
        self.startVideoStream(self.camera)


    #create directory if is not exist.
    def createPath(self,path):
        """ 
        Document will be added.
        """
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            return path
            

    #Yolo Models/Facial Landmarks
    def models(self):
        """ 
        Document will be added.
        """
        #dlib model
        face_landmarks=self.models_path+"shape_predictor_68_face_landmarks.dat"
        self.detector=dlib.get_frontal_face_detector()
        self.predictor=dlib.shape_predictor(face_landmarks)
        
        #eyes location index
        (self.l_start, self.l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.r_start, self.r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        #yolo model
        #yolov4_tiny->low accuracy, high fps
        #yolov4->high accuracy, low fps
        #self.net=cv2.dnn.readNet(self.models_path+"yolov4-tiny_training_last.weights",self.models_path+"yolov4-tiny_testing.cfg")
        self.net=cv2.dnn.readNet(self.models_path+"yolov4_training_last.weights",self.models_path+"yolov4_testing.cfg")

        #classes
        self.classes=["person","phone","smoke"]
        

    #threads start function
    def startThreads(self,_target,_args=()):
        """ 
        Document will be added.
        """
        t=Thread(target=_target,args=_args)
        t.daemon=True
        t.start()
        t.join()


    #Camera Run
    def startVideoStream(self,camera):
        """ 
        Document will be added.
        """
        time.sleep(2.0)#waiting for camera build up

        self.logFile("Camera Opened")#Camera Open Log
        
        while True:

            ret,self.frame=camera.read()#read camera 
            
            #if camera does not respond, shuts down system
            if not ret:
                break

            #resize frame
            self.frame = imutils.resize(self.frame, width=480,height=480)
            #grayscale frame
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            
            #if camera is blocked ->changeable
            if not self.gray.any():
                self.startThreads(self.controlCameraBlocked)

            if np.mean(self.gray)/255 < 0.5:
                self.histogramEqualization()

            
            #start object detection control, facial landmarks control and driver attention detection
            self.startThreads(self.objectDetection)
            self.startThreads(self.faceAndEyesDetection)

            self.startThreads(self.attentionDetection)
            self.startThreads(self.phoneDetection)
            self.startThreads(self.smokeDetection)

            #show camera
            cv2.imshow("Camera",self.frame)
            
            #press ESC or Q close camera
            key=cv2.waitKey(1) & 0xff
            if key==27 or key==ord("q"):
                break

        #stop processing
        self.stopVideoStream()


    def histogramEqualization(self):

        b_channel,g_channel,r_channel=np.dsplit(self.frame,3)
        b_channel=cv2.equalizeHist(b_channel)
        r_channel=cv2.equalizeHist(r_channel)
        g_channel=cv2.equalizeHist(g_channel)

        self.frame=np.dstack((b_channel,g_channel,r_channel))
        self.gray=cv2.equalizeHist(self.gray)        

    def controlCameraBlocked(self):
        """ 
        Document will be added.
        """
        #if camera blocked, when reach specified time, run warning and save image. 
        self.cover_counter+=1
        if self.cover_counter>10:
            time.sleep(1.0)
            self.errorTimeControl("Camera Blocked",5)
            self.warning("BlockedCameraWarning.mp3")
            #time.sleep(5.0)
        if self.gray.any():
            self.cover_counter=0


    #Yolo Object Detection
    def objectDetection(self):
        """ 
        Document will be added.
        """        
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
                if confidence>0.24:
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

        idx=cv2.dnn.NMSBoxes(boxes,confidences,0.24,0.4)
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
                cv2.rectangle(self.frame,(x,y),(x+w,y+h),color,1)
                cv2.putText(self.frame,label+confidence,(x,y+20),self.font,2,(255,255,255),2)
            #self.putTextVideoStream(label,confidence,x,y+10)
        except:
            pass


    #Calculate eye aspect ratio
    def findEyeAspectRatio(self,eye):
        """ 
        Document will be added.
        """
        first_height = dist.euclidean(eye[1], eye[5])
        second_height = dist.euclidean(eye[2], eye[4])
        eye_width = dist.euclidean(eye[0], eye[3])

        eye_aspect_ratio = (first_height + second_height) / (2.0 * eye_width)

        return eye_aspect_ratio

    #Face and Eye detection with dlib
    def faceAndEyesDetection(self):
        """ 
        Document will be added.
        """
        self.rects=self.detector(self.gray,0)

        for rect in self.rects:

            shape = self.predictor(self.gray, rect)
            shape = face_utils.shape_to_np(shape)


            left_eye = shape[self.l_start:self.l_end]
            right_eye = shape[self.r_start:self.r_end]

            left_ear = self.findEyeAspectRatio(left_eye)
            right_ear = self.findEyeAspectRatio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            self.drowsinessDetection(ear)
            self.putTextVideoStream("EAR",ear,250,30)


    #if driver look another direction long time, run warning and save image
    def attentionDetection(self):
        """ 
        Document will be added.
        """

        try:
            control=True if 0 in self.control_class_id else False
            print("control:",control)
            if not (not control or self.rects):
                self.attention_counter+=1
                print("attention:",self.attention_counter)
                if self.attention_counter>10:
                    self.errorTimeControl("attention",2)
                    self.warning("attentionWarning.mp3")
                    #time.sleep(5.0)
            else:
                self.attention_counter=0
        except:
            pass


    #if detect cigarette, run warning and save image
    def smokeDetection(self):
        """ 
        Document will be added.
        """
        try:
            control=True if 2 in self.control_class_id else False
            if control:
                self.smoke_counter+=1
                print("smoke:",self.smoke_counter)
                if self.smoke_counter>10:
                    self.errorTimeControl("Smoking",3)
                    self.warning("smokeWarning.mp3")
                    #time.sleep(5.0)
            else:
                self.smoke_counter=0
        except:
            pass


    #if detect phone, run warning and save image    
    def phoneDetection(self):
        """ 
        Document will be added.
        """
        try:
            control=True if 1 in self.control_class_id else False
            if control:
                self.phone_counter+=1
                print("Phone:",self.phone_counter)
                if self.phone_counter>=3:
                    self.errorTimeControl("Phone",4)
                    self.warning("PhoneWarning.mp3")
                    #time.sleep(5.0)
            else:
                self.phone_counter=0
        except:
            pass


    #if eyes aspect ratio < identified threshold. run warning and save image.
    def drowsinessDetection(self,ear):
        """ 
        Document will be added.
        """
        if ear < self.eyes_ar_threshold:
            self.drowsiness_counter += 1
            print("Drowsiness:",self.drowsiness_counter)
            if self.drowsiness_counter >= self.eye_ar_consec_frames:
                self.errorTimeControl("Drowsiness",1)
                self.warning("Drowsiness.mp3")
                #time.sleep(3.0)
        else:
            self.drowsiness_counter = 0


    #play warning sounds
    def warning(self,file):
        """ 
        Document will be added.
        """        
        
        path=self.alert_path+file
        playsound.playsound(path)
        time.sleep(2.0)


    #error time control, if error is same, must be wait 5(changeable) second save it.
    def errorTimeControl(self,error,error_code):
        """ 
        Document will be added.
        """        
        if error==self.last_err:
            if time.time()-self.last_err_time>5:
                self.saveImage(error,error_code)
        else:
            self.saveImage(error,error_code)


    #if detected any anomaly, save it.
    def saveImage(self,error,error_code):
        """ 
        Document will be added.
        """        
        self.last_err_time=time.time()

        img="{}_{}_{}.jpg".format(error_code,error,self.last_err_time)
        
        saved_img=self.save_image_path+img
        
        cv2.imwrite(saved_img,self.frame)
        
        self.logFile(error)
        self.err=error
        
        base64_image=self.imagetoBase64()
        
        self.jsonData(img,base64_image)


    #image to base64 format    
    def imagetoBase64(self):
        """
        Document will be added
        """

        flag,encoded_image=cv2.imencode(".jpg",self.frame)
        base64_image=base64.b64encode(encoded_image)
        base64_image=base64_image.decode("utf-8")
        return base64_image


    def jsonData(self,img,base64_image):
       
        img=img[:-4]#drop jpg extension
        
        data={img:base64_image}
        saved_path=self.save_image_path+img+".json"


        with open(saved_path, 'a') as outfile:
            json.dump(data, outfile)
    
    
    #logs
    def logFile(self,err):
        """ 
        Document will be added.
        """

        date=time.strftime("%x")
        _time=time.strftime("%X")
        with open("log.txt","a") as f:
            current_log="{} {} {}\n".format(date,_time,err)
            f.write(current_log)
        

    #put text camera screen, may be deleted
    def putTextVideoStream(self,text,value,x,y):
        cv2.putText(self.frame, text+ " : {:.3f}".format(value), (x, y),
        self.font, 2, (0, 0, 0), 2)
   

    #release camera, close camera window and log it.
    def stopVideoStream(self):
        """ 
        Document will be added.
        """
        self.logFile("Camera Closed")
        self.camera.release()
        cv2.destroyAllWindows()



if __name__=="__main__":
    driver=DriverSafety()
