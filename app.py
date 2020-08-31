from flask import Flask,jsonify,request,Response,render_template,redirect,url_for
from detection import DriverSafety
from threading import Lock,Thread
import time
import cv2


outputFrame = None
lock = Lock()

app=Flask(__name__)

def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route('/',methods=["GET"])
def detectDriver():
    global outputFrame,lock
    
    if request.method=="GET":
        args=request.args.get("video")

        if args=="0":
            args=int(args)
        driver=DriverSafety(args)

        time.sleep(1.0)

        #timer will be deleted.
        timer=0
        while timer<=10:
            driver.start_video_stream(driver.camera)
            timer+=1
            print("timer:",timer)
            with lock:
                outputFrame=driver.frame.copy()
        driver.stop_video_stream()
        
        return jsonify(driver.anomalies) 
        #redirect(url_for("summary",anomalies=driver.anomalies))

    elif request.method=="POST":
        pass

def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')



@app.route("/summary")
def summary():
    pass
    



@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__=="__main__":
    app.run(host="192.168.10.110",port=8080,debug=True)