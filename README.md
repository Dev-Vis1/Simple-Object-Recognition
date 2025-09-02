# Simple Object Recognition with PyTorch


This project demonstrates simple object recognition using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset with PyTorch. It also supports real-time object classification using your webcam.

## What is a CNN?
A Convolutional Neural Network (CNN) is a type of deep learning model especially effective for image recognition tasks. CNNs use layers of filters (convolutions) to automatically learn features from images, such as edges, shapes, and objects, making them well-suited for visual data.

## Features
- Trains a CNN on CIFAR-10 dataset (10 classes)
- Evaluates model accuracy
- Real-time webcam object recognition
- Demo class labels include: plane, car, bird, cat, deer, dog, frog, horse, ship, truck, mobile phone, pen

## Requirements
- Python 3.8+
- torch
- torchvision
- opencv-python

## Usage
1. Install dependencies:
   ```sh
   pip install torch torchvision opencv-python
   ```
2. Run the script:
   ```sh
   python simple_object_recognition.py
   ```
3. After training, a webcam window will open. Show an object to the camera to see the predicted class. Press 'q' to quit.

## Note
- The model is only trained on CIFAR-10 classes. The accuracy is quite low, 68% and some classes like 'mobile phone' and 'pen' are not in the training set.

## License
MIT
