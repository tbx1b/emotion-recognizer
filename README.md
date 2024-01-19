# Real-time Facial Emotion Recognition with TensorFlow and OpenCV

This project utilizes TensorFlow and OpenCV for real-time facial emotion recognition using a webcam. It also uses TensorFlow to train the AI model to reocognize emotions via the face using the FER-2013 dataset.

### Prerequisites

Make sure you have the required libraries installed. You can install them using:

```bash
pip install -r requirements.txt
```

Libraries included in requirements.txt:

    OpenCV: opencv-python==4.5.3.56
    TensorFlow: tensorflow==2.7.0
    NumPy: numpy==1.21.2
    Pandas: pandas==1.3.3

### Getting Started

This repository contains a pre-trained model for use in real-time facial recognition already, so to start you can just run the following command:

```bash
python3 facial_recognition.py
```

And a window will pop up with your camera feed and your emotion label.

### Usage

To train your own data set (Optimized for FER-2013), simply put it as 'dataset.csv' in the root of this project, and run the 'train_model.py' file.

### Project Structure

facial_recognition.py: The main script for real-time emotion recognition.
train_model.py: The script that handles TensorFlow and model training.
facial_expression_model.h5: Pre-trained emotion recognition model / output of training.
requirements.txt: List of Python dependencies.

### Acknowledgments

Face detection uses the Haar Cascade classifier provided by OpenCV.

Emotion recognition model is based and trained on TensorFlow.

### Additional Notes

If you encounter issues with the webcam, ensure that your camera is connected and accessible as the default webcam.

*This project is licensed under the MIT License - see the LICENSE.md file for details.*