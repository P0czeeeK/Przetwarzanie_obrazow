import cv2
import urllib.request
import numpy as np

url_up = "https://thumbs.dreamstime.com/b/wskazywanie-rki-emoji-ilustracja-kierunku-i-wskaz%C3%B3wek-z-%C3%B3t-rkawic-na-przezroczystoci-styl-rysowania-rk-palcem-404421919.jpg"
url_right = "https://png.pngtree.com/png-vector/20251111/ourmid/pngtree-pointing-finger-emoji-for-directional-and-interactive-designs-png-image_17915769.webp"
url_down = "https://symbl-cdn.com/i/webp/de/952ce9340428eb56a2b23f92bff20f.webp"
url_left = "https://i.pinimg.com/736x/6b/00/92/6b00925c9f8306e4c742cdb1b4297498.jpg"

resp = urllib.request.urlopen(url_up)
image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
image = cv2.imdecode(image, cv2.IMREAD_COLOR)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()