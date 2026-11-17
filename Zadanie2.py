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

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Zdalny obraz źródłowy")
plt.axis("off")

plt.subplot(1, 2, 2)
colors = ("r", "g", "b")
channel_names = ("Czerwony (R)", "Zielony (G)", "Niebieski (B)")

for i, col in enumerate(colors):
  hist = cv2.calcHist([img], [i], None, [256], [0, 256])
  plt.plot(hist, color=col, label=channel_names[i], alpha=0.8)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist_full = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.plot(hist_full, color="black", linestyle="--", label="Całość (Gray)", linewidth=1.5)

plt.title("Histogram dla kanałów i całego obrazu")
plt.xlabel("Wartość intensywności piksela (0 - 255)")
plt.ylabel("Liczba pikseli")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.show()