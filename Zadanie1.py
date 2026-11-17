import cv2
import matplotlib as plt
import numpy as np
import urllib.request

url = "https://thumb.wikimedia.org/wikipedia/commons/thumb/e/e4/Lemon.jpg/250px-Lemon.jpg?utm_source=commons.wikimedia.org&utm_campaign=parser&utm_content=thumbnail"

req_obj = urllib.request.Request(
    url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
)

with urllib.request.urlopen(req_obj) as req:
  arr = np.asarray(bytearray(req.read()), dtype=np.uint8)

img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

