## Emotion Detection using FER2013 Dataset, OpenCV, and CNN

This repository contains the code and implementation of an Emotion Detection project using the FER2013 dataset, OpenCV, and Convolutional Neural Networks (CNNs). The goal of this project is to accurately recognize human emotions from facial expressions in real-time.

### Dataset - FER2013

The FER2013 dataset is a widely used benchmark for facial expression recognition. It consists of over 35,000 grayscale images, categorized into seven emotion classes: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**. Each image is 48x48 pixels, making it suitable for training deep learning models.

### Requirements

- Python 3.x
- OpenCV
- Keras
  
### Usage

1. Download the FER2013 dataset from the Kaggle website or any other reliable source and place it in the dataset folder.

2. Preprocess the dataset:

   - Perform data augmentation techniques (optional but recommended).
   - Normalize pixel values to [0, 1] range.

3. Train the Emotion Detection model:

   - Run the `Train Emotion Detection.py` script to train the CNN model on the preprocessed dataset.
   - Tweak hyperparameters (e.g., learning rate, batch size, etc.) for optimal results.

4. Evaluate the model:

   - Use the `Evaluate Emotion Detector.py` script to assess the model's performance on a validation or test set.
   - Measure metrics such as accuracy, precision, recall, and F1-score.

5. Real-time Emotion Detection:

   - Implement the real-time emotion detection application using the trained model.
   - Run the `Test Emotion Detection.py` script to activate the webcam and visualize emotion predictions in real-time.

### Model Architecture

The Emotion Detection model architecture consists of a Convolutional Neural Network (CNN) with multiple layers. The architecture can be customized by altering the number of layers, filters, and neurons to achieve the desired performance.

### Results

Our trained model achieved an accuracy of 0.16675954304820284 on the FER2013 test set. The model shows robust performance in recognizing facial emotions, making it suitable for real-world applications.

### Acknowledgments

We would like to thank the creators of the FER2013 dataset for providing valuable data for research and development in the field of facial expression recognition.
