import cv2
import mediapipe as mp
import time
import numpy as np

cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

thumbPoints = []
wristPoints = []
thumb_and_wrist_angles = []

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for idH, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)

                if idH == 4:
                    thumbPoints.append([cx, cy])
                    
                if idH == 0:
                    wristPoints.append([cx, cy])

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    if thumbPoints and wristPoints is not None:
        print("thumb points", thumbPoints[0][1], thumbPoints[0][0])
        print("wrist points", wristPoints[0][1], wristPoints[0][0])
        angle_radian = np.arctan(abs((wristPoints[0][1]-thumbPoints[0][1])/(wristPoints[0][0]-thumbPoints[0][0])))
        angle_degree = (angle_radian*180)/3.14
        thumb_and_wrist_angles.append(angle_degree)
        print(f"Angles {thumb_and_wrist_angles}")
        thumbPoints.clear()
        wristPoints.clear()


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (18, 78), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
    

