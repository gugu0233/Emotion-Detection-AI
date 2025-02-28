import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

IMG_SIZE = (48, 48) 
BATCH_SIZE = 32 # procss 32 images at a time

# set up data augmentation rules
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True) # aply transformations to train data
val_datagen = ImageDataGenerator(rescale=1./255) # only rescale validation data

# load dataset and apply augmentation
train_generator = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    "dataset/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical"
)

# define cnn model
model = Sequential([
    #1st conv layer
    Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    
    #2nd conv layer
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    
    #3rd conv layer
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 emotions → softmax for classification
])

# compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# train model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# save Model
if not os.path.exists("model"):
    os.makedirs("model")

model.save("model/emotion_model.h5")
print("✅ Model saved successfully!")
