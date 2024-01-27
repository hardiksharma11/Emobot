# Face-Expression-Recognition-using-Deep-Learning
This project implements a convolutional neural network (CNN) to recognize facial expressions of seven different emotions: angry, disgust, fear, happy, neutral, sad, and surprise. The model is trained on the Face expression recognition dataset. Dataset E-link: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset.


Requirements:-
Python 3.11,
keras~=2.12.0rc0,
tensorflow,
numpy~=1.23.5,
matplotlib~=3.7.0,
pandas~=1.5.3,
seaborn,
opencv-contrib-python==4.7.0.68


Installation:-
1. First install the python.
2. After installing python. Install the packages listed in the requirements.txt. Use the command pip install -r requirements.txt 


Usage:-
1. Clone or download the repository.
2. Install the requirements
2. Run the main.py script using the following command:
   python main.py
3. The script will launch the webcam and start detecting emotions in real-time.


Files:-
1. main.py: This file is the entry point of the application. It loads the trained model and uses it to predict the emotions of faces in real-time using a webcam.
2. emotion_recognition_cnn.py: This file contains the Python code for building and training the CNN.
3. HaarcascadeclassifierCascadeClassifier.xml: The pre-trained Haar Cascade Classifier for detecting faces in images.
4. model.h5: The pre-trained Keras model for emotion detection.
