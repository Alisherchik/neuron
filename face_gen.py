import os
import cv2


path = os.path.dirname(os.path.abspath(__file__))
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

i = 0
offset = 50

face_number = input('Введите номер пользователя: ')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    clr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(clr, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        i += 1
        cv2.imwrite('dataset/face-'+face_number + '.' + str(i) + '.jpg', clr[y-offset:y+h+offset, x-offset:x+w+offset])
        cv2.rectangle(img, (x-50, y-50), (x+w+50, y-h+50), (225, 0, 0), 2)
        cv2.imshow("img", img[y-offset:y+h+offset, x-offset:x+w+offset])
        cv2.waitKey(100)

    if i > 30:
        cap.release()
        cv2.destroyAllWindows()
        break