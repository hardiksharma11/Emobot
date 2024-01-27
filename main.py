"""
This code is an implementation of a real-time facial emotion detection system using a Convolutional Neural
Network (CNN) in Python using the Keras deep learning library and OpenCV.

The program starts by importing the necessary libraries: numpy for numerical operations, cv2 for computer vision
and image processing, and Keras for loading the pre-trained CNN model.

The Haar cascade classifier CascadeClassifier is initialized using the pre-trained XML file
HaarcascadeclassifierCascadeClassifier.xml to detect frontal faces from the input frames.
Then, the pre-trained CNN model model.h5 is loaded using load_model() method from Keras.
This model is trained on the FER2013 dataset, which contains labeled images of seven emotions:
anger, disgust, fear, happiness, neutral, sadness, and surprise.

The program opens a connection to the default camera using cv2.VideoCapture(0) method. It then enters a
loop where it captures video frames and continuously detects facial emotions until the user stops the program.

For each video frame captured by the camera, the program first converts the image to grayscale using
cv2.cvtColor() method to reduce computational complexity. Then, the Haar cascade classifier is used to
detect all the faces in the image using face_classifier.detectMultiScale() method.

Once the faces are detected, the program loops through each detected face and extracts a region of interest
(ROI) that is the grayscale image of the face. The ROI is then resized to 48x48 pixels using cv2.resize()
method. This size is the same as the size of the input image for the CNN model.

Next, the ROI is preprocessed to be in the same format as the data used to train the CNN model.
It is normalized to a floating-point between 0 and 1 using roi_gray.astype('float')/255.0 method.
The ROI is then converted into a numpy array using img_to_array() method and finally reshaped to have a
4-dimensional shape with the first dimension being the batch size, which is set to 1 using np.expand_dims()
method.

The preprocessed ROI is then passed to the pre-trained CNN model for inference. The predict() method is used
to get the predicted probabilities for each of the seven emotion classes. The highest predicted probability
is selected and the corresponding emotion label is assigned to the face using
emotion_labels[prediction.argmax()].

Finally, the program draws a rectangle around the detected face and puts the predicted emotion label
next to it using cv2.rectangle() and cv2.putText() methods. If no faces are detected in the image, the
program simply displays the text 'No Faces' in the image using cv2.putText() method.

The resulting image is then displayed in a window using cv2.imshow() method. The program continues to process
frames until the user presses the 'q' key to quit the program. The camera is then released using cap.release()
method and all windows are closed using cv2.destroyAllWindows() method.

"""
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'HaarcascadeclassifierCascadeClassifier.xml')
classifier =load_model(r'model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()