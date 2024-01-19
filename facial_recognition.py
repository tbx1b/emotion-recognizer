import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('facial_expression_model.h5')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]

cap = cv2.VideoCapture(0)

x, y, w, h = 0, 0, 0, 0

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    emotion_label = "No Face Detected"
    emotion_color = (255, 255, 255)

    if len(faces) > 0:
        x, y, w, h = faces[0]

        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi / 255.0
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)

        predictions = model.predict(face_roi)
        emotion_label = emotion_labels[np.argmax(predictions)]
        emotion_color = emotion_colors[np.argmax(predictions)]

    cv2.putText(frame, emotion_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_color, 2, cv2.LINE_AA)

    cv2.rectangle(frame, (x, y), (x + w, y + h), emotion_color, 2)

    cv2.imshow('fer 2013 dataset', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
