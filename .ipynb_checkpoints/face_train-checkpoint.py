import cv2
import os
import numpy as np
from PIL import Image

# from dlib.python_examples.face_clustering import label
# from dlib.python_examples.face_jitter import image
# from dlib.tools.python.test.test_numpy_returns import image_path
# from face_gen import faces

path = os.path.dirname(os.path.abspath(__file__))
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
decoder = cv2.face.LBPHFaceRecognizer_create()

datapath = path + r'/dataset'


def get_images_and_labels(datapath):
    image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)] #список где кранится картинки

    image = []
    label = []

    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image_decoded = np.array(image_pil, 'uint8')
        user_id = int(os.path.split(image_path)[1].split('.')[0].replace('face-', ''))
        faces = classifier.detectMultiScale(image_decoded) # метод распознаёт лицо

        for (x, y, w, h) in faces:
            image.append(image_decoded[y: y + h, x: x + w])
            label.append(user_id)
            cv2.imshow('Sending faces to train', image_decoded[y: y + h, x: x + w])
            cv2.waitKey(100)

    return image, label

image, label = get_images_and_labels(datapath)
decoder.train(image, np.array(label))

decoder.save(path + r'/trainer/faces.xml')
cv2.destroyWindow()

