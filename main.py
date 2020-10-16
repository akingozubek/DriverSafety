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

    def __init__(self, camera=0, tiny: bool = True):

        # Thresholds, counters, timers and object x,y
        self.thresholds()
        self.counters()
        self.last_seen_timer()
        self.object_coordinate()

        # Camera and text font
        self.camera = cv2.VideoCapture(camera)
        self.font = cv2.FONT_HERSHEY_PLAIN

        # Create some directory
        self.alert_path = self.create_path("Sounds/")
        self.save_image_path = self.create_path("Images/")
        self.models_path = self.create_path("Models/")

        # Yolo and facial ladmarks models
        self.models(tiny)

        # Start camera
        self.start_video_stream(self.camera)


    # Threshold Variables
    def thresholds(self):

        # 930M Graphics Card -> Yolo-tiny 5.0~5.5 fps, yolo 0.7~0.8 fps
        self.EYES_AR_THRESHOLD = 0.24  # Eyes aspect ratio threshold
        self.EYE_AR_CONSEC_FRAMES = 25  # Drowsiness frames count
        self.OBJECT_CONSEC_FRAMES = 15  # Detect object frames count
        self.COVER_CONSEC_FRAMES = 25  # Cover camera frames count
        self.ATTENTION_CONSEC_FRAMES = 25  # Attenion detect frames count
        self.HAND_CONSEC_FRAMES = 60  # Hand frames count
        self.HIST_EQU_THRESHOLD = 0.3  # Histogram equalization threshold


    # Counters
    def counters(self):

        self.drowsiness_counter = 0
        self.cover_counter = 0
        self.attention_counter = 0
        self.smoke_counter = 0
        self.phone_counter = 0
        self.hand_counter = 0


    # Save the last anomalies detection time
    def last_seen_timer(self):
        self.drowsiness_timer = 0
        self.cover_timer = 0
        self.attention_timer = 0
        self.smoke_timer = 0
        self.phone_timer = 0
        self.hand_timer = 0


    # Draw object warning text coordinate
    def object_coordinate(self):
        self.smoke_x = 0
        self.smoke_y = 0
        self.phone_x = 0
        self.phone_y = 0


    # Create directory if is not exist.
    def create_path(self, path):

        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            return path


    # Yolo Models/Facial Landmarks
    def models(self, tiny):

        # Dlib model
        FACE_LANDMARKS = "{}shape_predictor_68_face_landmarks.dat".format(
            self.models_path)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(FACE_LANDMARKS)

        # Eyes location index
        (self.l_start,
         self.l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]

        (self.r_start,
         self.r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # Yolo model
        # Yolo_tiny->lower accuracy, higher fps
        # Yolo->higher accuracy, lower fps
        if tiny:
            self.net = cv2.dnn.readNet(
                self.models_path+"yolov4-tiny_training_last.weights",
                self.models_path+"yolov4-tiny_testing.cfg")

            self.net_hand = cv2.dnn.readNet(
                self.models_path+"yolo-tiny_hand.weights",
                self.models_path+"yolo-tiny_hand.cfg")

        else:
            self.net = cv2.dnn.readNet(
                self.models_path+"yolov4_training_last.weights",
                self.models_path+"yolov4_testing.cfg")

            self.net_hand = cv2.dnn.readNet(
                self.models_path+"yolo_hand.weights",
                self.models_path+"yolo_hand.cfg")

        # Classes
        self.classes = ("person", "phone", "smoke")


    # Threads start function
    def start_threads(self, target_, args_=()):

        t = Thread(target=target_, args=args_)
        t.daemon = True
        t.start()
        t.join()


    # Camera Run
    def start_video_stream(self, camera):

        time.sleep(2.0)  # Waiting for camera build up

        while True:

            ret, self.frame = camera.read()  # Read camera

            # if camera does not respond, shuts down system
            if not ret:
                break

            # Resize frame
            self.frame = imutils.resize(self.frame, width=480)

            self.height, self.width, c = self.frame.shape

            # Grayscale frame
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            # if camera is blocked
            if not self.gray.any():

                self.start_threads(self.camera_blocked_detection,
                                   args_=("CAMERA BLOCKED", 5))

            # if grayscale image is dark,
            # It is made brighter using Histogram Equalizer.
            if np.mean(self.gray)/255 < self.HIST_EQU_THRESHOLD:
                self.histogram_equalization()

            # Start object detection control,
            # Facial landmarks control and driver attention detection
            self.start_threads(self.object_detection,
                               args_=(self.net, self.classes,
                                      "object detect"))

            self.start_threads(self.object_detection,
                               args_=(self.net_hand, "hand",
                                      "hand detect"))

            self.start_threads(self.face_and_eyes_detection)

            self.start_threads(self.attention_detection,
                               args_=("ATTENTION", 2))

            self.start_threads(self.phone_detection, args_=("PHONE", 4))
            self.start_threads(self.smoke_detection, args_=("SMOKE", 3))
            self.start_threads(self.hand_detection)

            # Show camera
            cv2.imshow("Camera", self.frame)

            # Press ESC or Q close camera
            key = cv2.waitKey(1) & 0xff
            if key == 27 or key == ord("q"):
                break

        # Stop processing
        self.stop_video_stream()


    # Histogram equalization -> frame(blue,gray,red channels)
    # and grayscale frame.
    # if frame is dark, frame will be lighter.
    def histogram_equalization(self):

        # Divide blue,green,red channels
        b_ch, g_ch, r_ch = np.dsplit(self.frame, 3)

        # Histogram Equalization
        # For blue,green,red channels and grayscale frame.
        b_ch, g_ch, r_ch, self.gray = map(
            cv2.equalizeHist, [b_ch, g_ch, r_ch, self.gray])

        # Combine channels->frame.
        self.frame = np.dstack((b_ch, g_ch, r_ch))


    # Yolo Object Detection
    def object_detection(self, model, classes, type_):

        # Will be drawn box list, scores list and object id list
        boxes = []
        confidences = []
        class_ids = []

        # Image to blob and detect object
        blob = cv2.dnn.blobFromImage(
            self.frame, 1/255, (416, 416), (0, 0, 0),
            swapRB=True, crop=False)

        model.setInput(blob)

        out_layers_name = model.getUnconnectedOutLayersNames()
        layer_outs = model.forward(out_layers_name)

        # if there are any object
        for out in layer_outs:
            for detection in out:

                score = detection[5:]
                class_id = np.argmax(score)  # Object index
                confidence = score[class_id]  # Score is detected object

                # if score higher than threshold,
                # Draw rectangle object coordinates
                if confidence > 0.24:
                    center_x = int(detection[0]*self.width)
                    center_y = int(detection[0]*self.height)

                    w = int(detection[2]*self.width)
                    h = int(detection[2]*self.height)

                    x = int(center_x-w/2)
                    y = int(center_y-h/2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        if type_ == "object detect":

            # Use control object detection
            self.control_class_id = class_ids.copy()

        elif type_ == "hand detect":

            # Use control hand detection
            self.hand_class_id = class_ids.copy()

        idx = cv2.dnn.NMSBoxes(boxes, confidences, 0.24, 0.4)
        color = [0, 0, 255]

        # Show boxes and text
        try:
            for i in idx.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])

                #if detect object(smoke or phone), save its coordinates.
                if label == "smoke":
                    self.smoke_x = x
                    self.smoke_y = y+h
                elif label == "phone":
                    self.phone_x = x
                    self.phone_y = y+h

                confidence = round(confidences[i], 2)
                if label == "h":
                    cv2.rectangle(self.frame, (x, y+100), (x+w, (y+h)+100), color, 1)
                    self.put_text_video_stream(label, confidence, x, y+20)

        except:
            if type_ == "object detect":
                print("No Detect Object")

            elif type_ == "hand detect":
                print("No Detect Hand")


    # Calculate eye aspect ratio
    def find_eye_aspect_ratio(self, eye) -> float:

        first_height = dist.euclidean(eye[1], eye[5])
        second_height = dist.euclidean(eye[2], eye[4])

        eye_height = first_height + second_height
        eye_width = dist.euclidean(eye[0], eye[3])

        eye_aspect_ratio = eye_height / (2.0 * eye_width)

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

            self.drowsiness_detection(ear, rect, "DROWSINESS", 1)

            x, y = self.get_text_size("EAR", ear)
            self.put_text_video_stream("EAR", ear, self.width-100, y)


    # if eyes aspect ratio < identified threshold.
    # Run warning and save image.
    def drowsiness_detection(self, ear, rect, error_name, error_code):

        self.drowsiness_counter, self.drowsiness_timer = self.object_control(
            counter=self.drowsiness_counter,
            timer=self.drowsiness_timer,
            frame_threshold=self.EYE_AR_CONSEC_FRAMES,
            error_name=error_name,
            error_code=error_code,
            warning_name="DrowsinessWarning.mp3",
            controller=ear,
            time_limit=2,
            eyes_threshold=self.EYES_AR_THRESHOLD,
            x_coord=rect.left(),
            y_coord=rect.top()
        )


    # if driver look another direction long time,
    # Run warning and save image
    def attention_detection(self, error_name, error_code):

        x, y = self.get_text_size(error_name, error_code)
        x = round((self.width-x)/2)

        self.attention_counter, self.attention_timer = self.object_control(
            counter=self.attention_counter,
            timer=self.attention_timer,
            frame_threshold=self.ATTENTION_CONSEC_FRAMES,
            error_name=error_name,
            error_code=error_code,
            warning_name="AttentionWarning.mp3",
            controller=self.control_class_id,
            time_limit=2,
            class_id=0,
            x_coord=x,
            y_coord=y
        )


    # if detect smoke, run warning and save image
    def smoke_detection(self, error_name, error_code):

        self.smoke_counter, self.smoke_timer = self.object_control(
            counter=self.smoke_counter,
            timer=self.smoke_timer,
            frame_threshold=self.OBJECT_CONSEC_FRAMES,
            error_name=error_name,
            error_code=error_code,
            warning_name="SmokeWarning.mp3",
            controller=self.control_class_id,
            time_limit=5,
            class_id=2,
            x_coord=self.smoke_x,
            y_coord=self.smoke_y
        )


    # if detect phone, run warning and save image
    def phone_detection(self, error_name, error_code):

        self.phone_counter, self.phone_timer = self.object_control(
            counter=self.phone_counter,
            timer=self.phone_timer,
            frame_threshold=self.OBJECT_CONSEC_FRAMES,
            error_name=error_name,
            error_code=error_code,
            warning_name="PhoneWarning.mp3",
            controller=self.control_class_id,
            time_limit=5,
            class_id=1,
            x_coord=self.phone_x,
            y_coord=self.phone_y
        )


    # if hand detection, run warning(PHONE-SMOKE) and save image.
    def hand_detection(self):

        self.hand_counter, self.hand_timer = self.object_control(
            counter=self.hand_counter,
            timer=self.hand_timer,
            frame_threshold=self.HAND_CONSEC_FRAMES,
            error_name="PHONE",
            error_code=4,
            warning_name="PhoneWarning.mp3",
            controller=self.hand_class_id,
            time_limit=3,
            class_id=0,
            draw_text=False
        )


    # Control drowsiness, attention, smoke, phone and hand detection
    def object_control(self,
                       counter, timer, frame_threshold,
                       error_name, error_code, warning_name,
                       controller, time_limit=2, class_id=-1,
                       eyes_threshold=None,
                       draw_text=True, x_coord=0, y_coord=0) -> tuple:

        try:

            if error_code == 1:
                control = controller < eyes_threshold

            elif error_code == 2:
                condition = True if class_id in controller else False
                control = (condition and not self.rects)

            else:
                control = True if class_id in controller else False

            if control:
                counter += 1
                timer = time.time()

                if draw_text:
                    self.put_text_video_stream(error_name, error_code,
                                               x_coord, y_coord)

                if counter >= frame_threshold:
                    self.save_image(error_name, error_code)
                    self.warning(warning_name)
                    counter = 0

            else:
                if time.time() - timer > time_limit:
                    counter = 0

            return counter, timer

        except:
            print(error_name, "error")


    # if camera blocked, run warning and save image.
    def camera_blocked_detection(self, error_name, error_code):

        # if camera blocked, when reach specified time,
        # run warning and save image.
        self.cover_counter += 1

        (x, y) = self.get_text_size(error_name, error_code)

        self.put_text_video_stream(error_name, error_code, round(
            (self.width-x)/2), round((self.height-y)/2))

        if self.cover_counter > self.COVER_CONSEC_FRAMES:

            self.save_image(error_name, error_code)
            self.warning("BlockedCameraWarning.mp3")

            self.cover_counter = 0

        if self.gray.any():
            self.cover_counter = 0


    # Play warning sounds
    def warning(self, file):

        path = self.alert_path+file
        playsound.playsound(path)


    # if detected any anomaly, save jpg format.
    def save_image(self, error, error_code):

        error_time = time.time()

        img = "{}_{}_{}.jpg".format(error_code, error, error_time)

        saved_img = self.save_image_path+img

        cv2.imwrite(saved_img, self.frame)

        base64_image = self.image_to_base64()

        self.json_data(img, base64_image)


    # Image to base64 format
    def image_to_base64(self) -> str:

        flag, encoded_image = cv2.imencode(".jpg", self.frame)

        base64_image = base64.b64encode(encoded_image)
        base64_image = base64_image.decode("ascii")

        return base64_image


    # base64 to json
    def json_data(self, img, base64_image):

        img = img[:-4]  # Drop jpg extension

        data = {img: base64_image}
        saved_path = self.save_image_path+img+".json"

        with open(saved_path, 'a') as outfile:
            json.dump(data, outfile)


    # Put text camera screen
    def put_text_video_stream(self, text, value, x, y):

        if type(value) == int:

            cv2.putText(self.frame, "{} : {}".format(value, text),
                        (x, y), self.font, 1, (0, 0, 255), 2)

        else:

            cv2.putText(self.frame, "{} : {:.3f}".format(text, value),
                        (x, y), self.font, 1, (0, 0, 0), 2)


    # Find text size
    def get_text_size(self, text, value) -> tuple:
        (x, y), _ = cv2.getTextSize("{} : {}".format(value, text),
                                    self.font, 1, 2)

        return x, y


    # Release camera, close camera window
    def stop_video_stream(self) -> None:

        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    driver = DriverSafety(tiny=True)
