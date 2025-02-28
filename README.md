Emotion AI Detector:
-trains a CNN model on facial expression images
-uses OpenCV to detect faces in real-time
-predicts emoton from the detected face using the trained model
-sisplays results by drawing a rectangle around the face and showing the predicted emotion

Used the FER2013 dataset from Kaggle: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data:
dataset/
│── train/            # used to train the CNN model
│   ├── angry/
│   ├── happy/
│   ├── sad/
│   ├── neutral/
│   ├── surprised/
│── test/             # used to evaluate the model after training
│   ├── angry/
│   ├── happy/
│   ├── sad/
│   ├── neutral/
│   ├── surprised/

