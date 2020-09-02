import base64
import json

import requests

param = {'video': 0}
response = requests.get("http://192.168.10.110:8080/detection", params=param)
json_data = response.json()

for k, v in json_data.items():
    with open(f"Images/{k}.jpg", "wb") as f:
        byte_value = str.encode(v, "ascii")
        f.write(base64.decodebytes(byte_value))
