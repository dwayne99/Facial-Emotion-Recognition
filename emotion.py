import os
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

# Load the model 
model = tf.keras.models.load_model('FED_model.h5')

# Defining the classes 
classes  = {
    0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy',
    4:'Sad', 5:'Surprise', 6:'Neutral'
}

# Load the Haar cascade
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)

while True:
    # captures the frame and returns a boolean value and image
    face, test_img = capture.read()
    if not face:
        continue
    # Converting our image to grayscale
    gray_image = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Detecting the face using the Haar Cascade Filter
    detected_face = haar_cascade.detectMultiScale(gray_image,1.32,5)

    for (x,y,w,h) in detected_face:
        
        # Drawing a rectangle on the detected face
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
       
        # Cropping the face
        cropped_face = gray_image[y:y+h,x:x+w]

        # Resizing the cropped face
        cropped_face = cv2.resize(cropped_face,(48,48))

        # Image to array
        cropped_face_px = image.img_to_array(cropped_face)

        # Expanding image to match input shape of model
        cropped_face_px = np.expand_dims(cropped_face_px, axis=0)

        # Normalizing 
        cropped_face_px /= 255
        
        # Predicting with the model 
        predictions = model.predict(cropped_face_px)

        # Finding Max of predictions
        max_index = np.argmax(predictions[0])
        
        # Mapping index to emotion
        emotion = classes[max_index]

        # Display onto the screen
        cv2.putText(test_img, emotion, (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


    resized_img = cv2.resize(test_img, (1000,700))
    cv2.imshow('Facial Emotion Recognition',resized_img)

    if cv2.waitKey(10) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows



        
        
