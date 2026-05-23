import cv2
import mediapipe as mp
import random

# Kamera aç
kamera = cv2.VideoCapture(0)

# El algılama sistemi
mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils

# Rastgele kalp yerleri
kalpler = []

for i in range(20):
    x = random.randint(50, 550)
    y = random.randint(50, 400)
    kalpler.append((x, y))

while True:
    success, img = kamera.read()

    if not success:
        break

    # Aynalama
    img = cv2.flip(img, 1)

    # RGB çevir
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # El algılama
    results = hands.process(imgRGB)

    el_var = False

    if results.multi_hand_landmarks:
        el_var = True

        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(
                img,
                handLms,
                mpHands.HAND_CONNECTIONS
            )

    # El varsa kalpler göster
    if el_var:
        for (x, y) in kalpler:
            cv2.putText(
                img,
                "❤",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3
            )

    cv2.imshow("Kamera", img)

    # ESC ile çık
    if cv2.waitKey(1) == 27:
        break

kamera.release()
cv2.destroyAllWindows()