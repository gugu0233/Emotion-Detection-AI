import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("model/emotion_model.h5")
class_labels = ["happy", "sad", "angry", "neutral", "surprised"]

# Load Haarcascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, axis=0) / 255.0
        
        predictions = model.predict(face)
        emotion = class_labels[np.argmax(predictions)]

        # Draw text and rectangle
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Emotion Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press "q" to quit
        break

cap.release()
cv2.destroyAllWindows()
