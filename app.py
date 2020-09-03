import os
import time
from threading import Thread

import cv2
from flask import Flask, Response, jsonify, redirect, request, url_for, render_template
from waitress import serve
from werkzeug.utils import secure_filename

from detection import DriverSafety

outputFrame = None


UPLOAD_DIRECTORY = "Uploads"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.mkdir(UPLOAD_DIRECTORY)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIRECTORY


ALLOWED_EXTENSION = {"mp4", "avi", "wmv"}


def allowed_extension(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSION


@app.route("/", methods=["GET", "POST"])
def main():

    if request.method == "POST":
        if 'file' not in request.files:
            response = jsonify({'message': 'No file part in the request'})
            response.status_code = 400
            return response

        file = request.files["file"]

        if file.filename == "":
            response = jsonify({'message': 'No file selected for uploading'})
            response.status_code = 400
            return response

        if file and allowed_extension(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            return redirect(url_for("detectDriver", video=file_path))

        else:
            response = jsonify(
                {'message': 'Allowed file types are mp4, avi, wmv'})
            response.status_code = 400
            return response


@app.route('/detection', methods=["GET"])
def detectDriver():
    global outputFrame

    if request.method == "GET":
        args = request.args.get("video")

        driver = DriverSafety(args)

        while True:
            ret = driver.start_video_stream(driver.camera)

            if not ret:
                break

            outputFrame = driver.frame.copy()
        driver.stop_video_stream()
        os.remove(args)
        return jsonify(driver.anomalies)


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame

    # loop over frames from the output stream
    while True:
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


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    #app.run(host="192.168.10.110", port=8080, debug=True)
    serve(app, host="localhost", port=8080)
