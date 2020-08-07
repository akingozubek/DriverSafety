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


#Sesler->Sounds Klasöründe
#Modeller->Models Klasöründe
#Resimler->Images Klasörüne

class DriverSafety():
    def __init__(self,queueSize=128,camera=0):
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 40
        self.COUNTER=0
        self.COVER_COUNTER=0
        self.ATTENTION_COUNTER=0
        self.camera=cv2.VideoCapture(camera)
        self.last_err=""
        self.last_err_time=0

        self.alert_path="Sounds/"
        self.save_image_path="Images/"
        self.models_path="Models/"


        self.models()
        self.facialLandmarksIDX()
        self.readObjectClassNames()

        self.startVideoStream(self.camera)

    def warning(self,file):
        path=self.alert_path+file
        playsound.playsound(path)

    def models(self):
        self.face_landmarks=self.models_path+"shape_predictor_68_face_landmarks.dat"
        self.detector=dlib.get_frontal_face_detector()
        self.predictor=dlib.shape_predictor(self.face_landmarks)
        self.net=cv2.dnn.readNet(self.models_path+"yolov3.weights",self.models_path+"yolov3.cfg")

    def facialLandmarksIDX(self):
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def startVideoStream(self,camera):
        time.sleep(2.0)
        self.logFile("Camera Opened")
        while True:
            self._time=time.time()
            ret,self.frame=camera.read()
            if not ret:
                break
            self.frame = imutils.resize(self.frame, width=600,height=800)
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            self.frame=np.dstack([self.gray,self.gray,self.gray])
            if not self.gray.any():
                self.t1=Thread(target=self.controlCameraBlocked)
                self.t1.daemon=True
                self.t1.start()
                self.t1.join()
 
            

            self.t3=Thread(target=self.objectDetection)
            self.t4=Thread(target=self.faceAndEyesDetection)
            self.t5=Thread(target=self.attentionDetection)

            self.t3.daemon=True
            self.t4.daemon=True
            self.t5.daemon=True

            self.t3.start()
            self.t4.start()
            self.t5.start()

            self.t3.join()
            self.t4.join()
            self.t5.join()


            cv2.imshow("Camera",self.frame)
            self.key=cv2.waitKey(1) & 0xff
            if self.key==27 or self.key==ord("q"):
                break
        self.stopVideoStream()
    
    def putTextVideoStream(self,text,value,x,y):
        cv2.putText(self.frame, text+ " : {:.3f}".format(value), (x, y),
        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    
    

    def controlCameraBlocked(self):
        self.COVER_COUNTER+=1
        if self.COVER_COUNTER>5:
            time.sleep(1.0)
            self.warning("BlockedCameraWarning.mp3")
            self.saveImage(self.frame,"Camera Blocked")
        if self.gray.any():
            self.COVER_COUNTER=0
            
    def readObjectClassNames(self):
        with open(self.models_path+"coco.names","r") as f:
            self.classes=f.read().splitlines()
        
    def objectDetection(self):

        height,width=self.frame.shape[0],self.frame.shape[1]
        boxes=[]
        confidences=[]
        class_ids=[]

        blob=cv2.dnn.blobFromImage(self.frame,1/255,(416,416),(0,0,0),swapRB=True,crop=False)
        self.net.setInput(blob)
        out_layers_name=self.net.getUnconnectedOutLayersNames()
        layer_outs=self.net.forward(out_layers_name)



        for out in layer_outs:
            for detection in out:
                score=detection[5:]
                class_id=np.argmax(score)
                self.control_class_id=class_id.copy()
                confidence=score[class_id]
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
        font=cv2.FONT_HERSHEY_PLAIN
        colors=np.random.uniform(0,255,size=(len(boxes),3))
        try:
            for i in idx.flatten():
                x,y,w,h=boxes[i]
                label=str(self.classes[class_ids[i]])
                confidence=str(round(confidences[i],2))
                color=colors[i]
                cv2.rectangle(self.frame,(x,y),(x+w,y+h),color,1)
                #cv2.putText(self.frame,label+" : "+confidence,(x,y+20),font,2,(255,255,255),2)
                self.putTextVideoStream(label,confidence,x,y+10)
        except:
            pass

    def attentionDetection(self):
        try:
            if not self.rects and self.control_class_id==0:
                self.ATTENTION_COUNTER+=1
                if self.ATTENTION_COUNTER>10:
                    self.warning("AttentionWarning.mp3")
                    time.sleep(1.0)
            else:
                self.ATTENTION_COUNTER=0

        except:
            pass

    def smokeDetection(self):
        pass
    
    def phoneDetection(self):
        pass
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

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            self.drowsinessDetection(ear)
            self.putTextVideoStream("EAR",ear,300,30)

    def findEyeAspectRatio(self,eye):

        first_height = dist.euclidean(eye[1], eye[5])
        second_height = dist.euclidean(eye[2], eye[4])
        eye_width = dist.euclidean(eye[0], eye[3])

        self.eye_aspect_ratio = (first_height + second_height) / (2.0 * eye_width)

        return self.eye_aspect_ratio

    def drowsinessDetection(self,ear):
 
        if ear < self.EYE_AR_THRESH:
            self.COUNTER += 1

            if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                self.saveImage(self.frame,"drowsiness")
                t2 = Thread(target=self.warning,
                    args=("drowsiness.mp3",))
                t2.daemon = True
                t2.start()
                time.sleep(3.0)
        else:
            self.COUNTER = 0

    def timeCounter(self):
        pass

    def saveImage(self,frame,err):
        print("last_Err:",self.last_err)
        print("err:",err)
        print("_time:",self._time)
        print("last_err_time:",self.last_err_time)
        if err==self.last_err:
            if self._time-self.last_err_time>5:
                frame=frame[:,:,0]
                frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
                cv2.imwrite(self.save_image_path+err+".jpg",frame)
        else:
            frame=frame[:,:,0]
            frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
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
        
    
    def createSound(self,text,filename):
        voice=gTTS(text,lang="en")
        voice.save("Sounds/"+filename)

    def stopVideoStream(self):
        self.logFile("Camera Closed")
        self.camera.release()
        cv2.destroyAllWindows()




if __name__=="__main__":
    driver=DriverSafety()
