import argparse
import base64
import json

import requests

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True,
                help="path to file")

args = vars(ap.parse_args())

param = {'file': open(args["file"], "rb")}
response = requests.post("http://192.168.10.110:8080/videofile", files=param)
json_data = response.json()

for k, v in json_data.items():
    with open(f"Images/{k}.jpg", "wb") as f:
        byte_value = str.encode(v, "ascii")
        f.write(base64.decodebytes(byte_value))
