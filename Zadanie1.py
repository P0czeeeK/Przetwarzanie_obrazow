import cv2
import matplotlib.pyplot as plt
import numpy as np
import urllib.request

url = "https://thumb.wikimedia.org/wikipedia/commons/thumb/e/e4/Lemon.jpg/250px-Lemon.jpg?utm_source=commons.wikimedia.org&utm_campaign=parser&utm_content=thumbnail"

req_obj = urllib.request.Request(
    url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
)

with urllib.request.urlopen(req_obj) as req:
  arr = np.asarray(bytearray(req.read()), dtype=np.uint8)

img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(5, 5))
plt.imshow(img_rgb)
plt.title("Oryginalny obraz")
plt.axis("off")
plt.show()

width = int(img.shape[1] * 0.5)
height = int(img.shape[0] * 0.5)

img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

img_rotated = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)

plt.figure(figsize=(5, 5))
plt.imshow(img_rotated, cmap="gray")
plt.title("Obraz po resize, grayscale i obrocie o 90°")
plt.axis("off")
plt.show()

print(
    "Macierz obrazu:\n"
)
print(img_rotated)
print(
    f"\nPełne wymiary macierzy wynikowej: {img_rotated.shape} (wysokość, szerokość)"
)