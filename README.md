# DriverSafety

DriverSafety works in real time. It is about driver sleep and attention control, phone and smoking detection. It also warns about these situations and saves the image.

Project also has hand detection, but this feature is still under development. You can remove this feature from the project.

Project has YOLO and dlib models. if you want to use project, you must install darknet, dlib and opencv.
<https://pjreddie.com/darknet/install/>

<http://dlib.net/compile.html>

<https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html>

<https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html>

Project works main.py file, also you can use app.py for API.

if you receive `RuntimeError: Error deserializing object of type int`,
you must download `shap_predictor_68_face_landmarks.dat` and it must be in **Models** directory.
You can download use this command:

```Shell
wget -O shape_predictor_68_face_landmarks.dat https://media.githubusercontent.com/media/akingozubek/DriverSafety/master/Models/shape_predictor_68_face_landmarks.dat?token=AHWAIWC2QWB7QLHNJQDROS27RYTJG
```


Project works yolo-tiny models, if you want more accuracy you can download and use yolo models from this link:
<https://drive.google.com/drive/folders/1XO3oXR1bhZBTK8fFoRNGmzUq7AZ3WFfp?usp=sharing>

if you have any suggestion or question you can ask me.

I can answer your question as far as I know.

# License

DriverSafety is MIT-licensed.
