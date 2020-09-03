# import the necessary packages
import base64
import json
import os
import time
from threading import Thread

import cv2
import dlib
import imutils
import numpy as np
import playsound
from imutils import face_utils
from scipy.spatial import distance as dist


class DriverSafety():

    def __init__(self, camera=0):

        # Threshold Variables
        # yolo-tiny 5.0~5.5 fps, yolo 0.7~0.8 fps
        self.EYES_AR_THRESHOLD = 0.24  # Eyes aspect ratio threshold
        self.EYE_AR_CONSEC_FRAMES = 25  # drowsiness frames count
        self.OBJECT_CONSEC_FRAMES = 15  # detect object frames count
        self.COVER_CONSEC_FRAMES = 25  # cover camera frames count
        self.ATTENTION_CONSEC_FRAMES = 30  # attenion detect frames count
        self.HIST_EQU_THRESHOLD = 0.3  # histogram equalization threshold

        # Counters
        self.drowsiness_counter = 0
        self.cover_counter = 0
        self.attention_counter = 0
        self.smoke_counter = 0
        self.phone_counter = 0

        # camera and text font
        self.camera = cv2.VideoCapture(camera)
        self.font = cv2.FONT_HERSHEY_PLAIN

        # log and image save adaptibility
        self.last_err = ""
        self.last_err_time = 0

        # for saving all anomalies run time.
        self.anomalies = dict()

        # Create some directory
        self.alert_path = self.create_path("Sounds/")
        self.models_path = self.create_path("Models/")

        # yolo models/facial ladmarks models
        self.models()

    # create directory if is not exist.

    def create_path(self, path):

        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            return path

    # Yolo Models/Facial Landmarks

    def models(self):

        # dlib model
        FACE_LANDMARKS = self.models_path+"shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(FACE_LANDMARKS)

        # eyes location index
        (self.l_start,
         self.l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]

        (self.r_start,
         self.r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # yolo model
        # yolov4_tiny->low accuracy, high fps
        # yolov4->high accuracy, low fps

        self.net = cv2.dnn.readNet(
            self.models_path+"yolov4-tiny_training_last.weights",
            self.models_path+"yolov4-tiny_testing.cfg"
        )

        # self.net = cv2.dnn.readNet(
        #    self.models_path+"yolov4_training_last.weights",
        #    self.models_path+"yolov4_testing.cfg"
        # )

        # classes
        self.classes = ("person", "phone", "smoke")

    # threads start function

    def start_threads(self, target_, args_=()):

        t = Thread(target=target_, args=args_)
        t.daemon = True
        t.start()
        t.join()

    # Camera Run

    def start_video_stream(self, camera):

        ret, self.frame = camera.read()  # read camera

        if not ret:
            return ret

        # resize frame
        self.frame = imutils.resize(self.frame, width=480, height=480)

        # grayscale frame
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # if camera is blocked
        if not self.gray.any():
            self.start_threads(self.camera_blocked_detection)

        # if grayscale image is dark, it is made brighter using Histogram Equalizer.
        if np.mean(self.gray)/255 < self.HIST_EQU_THRESHOLD:
            self.histogram_equalization()

        # start object detection control, facial landmarks control and driver attention detection
        self.start_threads(self.object_detection)
        self.start_threads(self.face_and_eyes_detection)

        self.start_threads(self.attention_detection)
        self.start_threads(self.phone_detection)
        self.start_threads(self.smoke_detection)

        return ret

    # histogram equalization -> frame(blue,gray,red channels) and grayscale frame.

    def histogram_equalization(self):

        # divide blue,green,red channels
        b_ch, g_ch, r_ch = np.dsplit(self.frame, 3)

        # Histogram Equalization, blue,green,red channels and grayscale frame.
        b_ch, g_ch, r_ch, self.gray = map(
            cv2.equalizeHist, [b_ch, g_ch, r_ch, self.gray])

        # combine channels->frame.
        self.frame = np.dstack((b_ch, g_ch, r_ch))

    # control camera is blocked.

    def camera_blocked_detection(self):

        # if camera blocked, when reach specified time, run warning and save image.
        self.cover_counter += 1

        # self.attention_counter=0->if using tiny. bug.
        if self.cover_counter > self.COVER_CONSEC_FRAMES:
            self.error_time_control("Camera Blocked", 5)
            self.warning("BlockedCameraWarning.mp3")
            self.cover_counter = 0

        if self.gray.any():
            self.cover_counter = 0

    # Yolo Object Detection

    def object_detection(self):

        height, width, _ = self.frame.shape

        # will be drawn box list, scores list and object id list
        boxes = []
        confidences = []
        class_ids = []

        # image to blob and detect object
        blob = cv2.dnn.blobFromImage(
            self.frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        self.net.setInput(blob)

        out_layers_name = self.net.getUnconnectedOutLayersNames()
        layer_outs = self.net.forward(out_layers_name)

        # if there are any object
        for out in layer_outs:
            for detection in out:
                score = detection[5:]
                class_id = np.argmax(score)  # object index
                confidence = score[class_id]  # score is detected object

                # if score %50 coordination and boxes process
                if confidence > 0.24:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[0]*height)

                    w = int(detection[2]*width)
                    h = int(detection[2]*height)

                    x = int(center_x-w/2)
                    y = int(center_y-h/2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # use to control object detection
        self.control_class_id = class_ids.copy()

        idx = cv2.dnn.NMSBoxes(boxes, confidences, 0.24, 0.4)
        color = [0, 0, 255]

        # show boxes and text
        # try:
        try:
            for i in idx.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                # print(label)
                confidence = str(round(confidences[i], 2))
                cv2.rectangle(self.frame, (x, y), (x+w, y+h), color, 1)
                cv2.putText(self.frame, label+confidence, (x, y+20),
                            self.font, 2, (255, 255, 255), 2)
        except:
            pass

    # Calculate eye aspect ratio

    def find_eye_aspect_ratio(self, eye):

        first_height = dist.euclidean(eye[1], eye[5])
        second_height = dist.euclidean(eye[2], eye[4])
        eye_width = dist.euclidean(eye[0], eye[3])

        eye_aspect_ratio = (first_height + second_height) / (2.0 * eye_width)

        return eye_aspect_ratio

    # Face and Eye detection with dlib

    def face_and_eyes_detection(self):

        self.rects = self.detector(self.gray, 0)

        for rect in self.rects:

            shape = self.predictor(self.gray, rect)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[self.l_start:self.l_end]
            right_eye = shape[self.r_start:self.r_end]

            left_ear = self.find_eye_aspect_ratio(left_eye)
            right_ear = self.find_eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            self.drowsiness_detection(ear)

    # if driver look another direction long time, run warning and save image

    def attention_detection(self):

        try:
            control = True if 0 in self.control_class_id else False

            if not (not control or self.rects):
                self.attention_counter += 1

                if self.attention_counter > self.ATTENTION_CONSEC_FRAMES:
                    self.error_time_control("Attention", 2)
                    self.warning("attentionWarning.mp3")
                    self.attention_counter = 0

            else:
                self.attention_counter = 0

        except:
            pass

    # if detect cigarette, run warning and save image

    def smoke_detection(self):

        self.smoke_counter = self.object_control(
            2, self.smoke_counter, "Smoke", 3, "smokeWarning.mp3")

    # if detect phone, run warning and save image

    def phone_detection(self):

        self.phone_counter = self.object_control(
            1, self.phone_counter, "Phone", 4, "phoneWarning.mp3")

    # control smoke and phone

    def object_control(self, class_id, counter, error, error_code, warning_name):
        try:
            control = True if class_id in self.control_class_id else False

            if control:
                counter += 1

                if counter >= self.OBJECT_CONSEC_FRAMES:
                    self.error_time_control(error, error_code)
                    self.warning(warning_name)
                    counter = 0

            else:
                counter = 0
            return counter

        except:
            return counter

    # if eyes aspect ratio < identified threshold. run warning and save image.

    def drowsiness_detection(self, ear):

        if ear < self.EYES_AR_THRESHOLD:
            self.drowsiness_counter += 1

            if self.drowsiness_counter >= self.EYE_AR_CONSEC_FRAMES:
                self.error_time_control("Drowsiness", 1)
                self.warning("DrowsinessWarning.mp3")
                self.drowsiness_counter = 0

        else:
            self.drowsiness_counter = 0

    # play warning sounds

    def warning(self, file):

        path = self.alert_path+file
        playsound.playsound(path)

    # error time control, if error is same, must be wait 5(changeable) second save it.

    def error_time_control(self, error, error_code):

        if error == self.last_err:
            if time.time()-self.last_err_time > 5:
                self.save_image(error, error_code)
        else:
            self.save_image(error, error_code)

    # if detected any anomaly, save it.

    def save_image(self, error, error_code):

        self.last_err_time = time.time()

        img = "{}_{}_{}.jpg".format(error_code, error, self.last_err_time)

        self.err = error

        base64_image = self.image_to_base64()

        self.json_data(img, base64_image)

    # image to base64 format

    def image_to_base64(self):

        flag, encoded_image = cv2.imencode(".jpg", self.frame)
        base64_image = base64.b64encode(encoded_image)
        base64_image = base64_image.decode("ascii")
        return base64_image

    def json_data(self, img, base64_image):

        img = img[:-4]  # drop jpg extension

        self.anomalies[img] = base64_image


    def stop_video_stream(self):

        try:
            self.camera.release()
        except:
            pass
        finally:
            self.camera.release()


if __name__ == "__main__":
    driver = DriverSafety()
