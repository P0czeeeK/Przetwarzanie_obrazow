#Aby przetestować różne kierunki należy zmienić url w linijce 19

import cv2
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

url_up = "Zdjecie1.jpg"
url_right = "Zdjecie4.jpg"
url_down = "Zdjecie2.jpg"
url_left = "Zdjecie3.jpg"

# resp = urllib.request.urlopen(url_left)
# image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
# image = cv2.imdecode(image, cv2.IMREAD_COLOR)

image = cv2.imread(url_left)

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

    cv2.putText(
        small_image,
        direction,
        (50,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

else:
    print("Nie wykryto dłoni")


cv2.imshow("Image", small_image)
cv2.waitKey(0)
cv2.destroyAllWindows()