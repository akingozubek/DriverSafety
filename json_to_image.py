import base64
import json

import requests

param = {'video': 'smoke-deneme.mp4'}
param = {'video': '0'}
response = requests.get("http://localhost:8080/", params=param)
json_data = response.json()

for k, v in json_data.items():
    with open(k+".jpg", "wb") as f:
        byte_value = str.encode(v, "ascii")
        f.write(base64.decodebytes(byte_value))
