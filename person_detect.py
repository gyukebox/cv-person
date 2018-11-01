import cv2

face_default = "haarcascade_frontalface_default.xml"
faceCascade_default = cv2.CascadeClassifier(face_default)
video_capture = cv2.VideoCapture(0)

anterior1 = 0
cnt1 = 0

CAPTURED_FRAME = 0

while CAPTURED_FRAME < 10:
    if not video_capture.isOpened():
        raise EnvironmentError('Camera not available')

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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if anterior1 != len(faces1):
        anterior1 = len(faces1)
        cnt1 = cnt1 + 1

        if cnt1 > 1:
            print("-------------face detected by default!")
            CAPTURED_FRAME += 1
            cnt1 = 0

# When everything is done, release the capture
# TODO change this with server-send code: Integrate with total recognition
video_capture.release()
cv2.destroyAllWindows()
