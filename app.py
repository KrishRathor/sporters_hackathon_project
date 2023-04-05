from flask import Flask
from flask import render_template, Response
import cv2
import time
import numpy as np
import mediapipe as mp

app = Flask(__name__)

cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

thumbPoints = []
wristPoints = []
thumb_and_wrist_angles = []

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for idH, lm in enumerate(handLms.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)

                        if idH == 4:
                            thumbPoints.append([cx, cy])
                            
                        if idH == 0:
                            wristPoints.append([cx, cy])

                    mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
            
            
        frame = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def product_page():
    return render_template('product_page.html')

@app.route("/video")
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(debug=True)
