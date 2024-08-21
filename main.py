import cv2

faceClassifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

videoCapture = cv2.VideoCapture(0)

def detectBoundingBox(vid):
    greyImage = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(greyImage, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x+w, y+h), (0, 255, 0), 4)
    return faces

while True:
    result, videoFrame = videoCapture.read()
    if result is False:
        break
    faces = detectBoundingBox(videoFrame)
    cv2.imshow('Face Detection', videoFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()