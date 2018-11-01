import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

face_default = "haarcascade_frontalface_default.xml"

faceCascade_default = cv2.CascadeClassifier(face_default)


# log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)

anterior1 = 0
cnt1 = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces1 = faceCascade_default.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces1:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    if anterior1 != len(faces1):
        anterior1 = len(faces1)
        # log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
        cnt1 = cnt1+1

        if cnt1 > 1: 
            print("-------------face detected by default!")
            cnt1 = 0

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
