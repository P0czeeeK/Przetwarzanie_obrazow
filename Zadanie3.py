import cv2
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

url_up = "Zdjecie1.jpg"
url_right = "https://png.pngtree.com/png-vector/20251111/ourmid/pngtree-pointing-finger-emoji-for-directional-and-interactive-designs-png-image_17915769.webp"
url_down = "https://symbl-cdn.com/i/webp/de/952ce9340428eb56a2b23f92bff20f.webp"
url_left = "https://assets.streamlinehq.com/image/private/w_512,h_512,ar_1/f_auto/v1/icons/4/pointing-left-emoji-sg240l8ioed4zkd4e04447.png/pointing-left-emoji-9ovxwwvyp2i4b6l4ojjo7.png?_a=DATAiZAAZAA0"

# resp = urllib.request.urlopen(url_left)
# image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
# image = cv2.imdecode(image, cv2.IMREAD_COLOR)

image = cv2.imread(url_up)

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb
)

small_image = cv2.resize(
    image,
    None,
    fx=0.3,
    fy=0.3
)



options = vision.HandLandmarkerOptions(
    base_options = BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    num_hands = 1
)

detector = vision.HandLandmarker.create_from_options(options)

result = detector.detect(mp_image)

if result.hand_landmarks:

    hand = result.hand_landmarks[0]

    mcp = hand[5]
    tip = hand[8]

    dx = tip.x - mcp.x
    dy = tip.y - mcp.y

    if abs(dx) > abs(dy):
        direction = "PRAWO" if dx > 0 else "LEWO"
    else:
        direction = "DOL" if dy > 0 else "GORA"

    print("Kierunek: ", direction)

else:
    print("Nie wykryto dłoni")


cv2.imshow("Image", small_image)
cv2.waitKey(0)
cv2.destroyAllWindows()