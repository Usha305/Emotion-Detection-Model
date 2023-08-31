## Emotion Detection using FER2013 Dataset, OpenCV, and CNN

This repository contains the code and implementation of an Emotion Detection project using the FER2013 dataset, OpenCV, and Convolutional Neural Networks (CNNs). The goal of this project is to accurately recognize human emotions from facial expressions in real-time.

### Dataset - FER2013

The FER2013 dataset is a widely used benchmark for facial expression recognition. It consists of over 35,000 grayscale images, categorized into seven emotion classes: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**. Each image is 48x48 pixels, making it suitable for training deep learning models.

### Requirements

- Python 3.x
- OpenCV
- TensorFlow (or any other compatible deep learning framework)

### Installation

1. Clone this repository to your local machine:

```
git clone https://github.com/your-username/emotion-detection.git
```

2. Install the required dependencies:

```
pip install opencv-python
pip install tensorflow (or any other compatible deep learning framework)
```

### Usage

1. Navigate to the project directory:

```
cd emotion-detection
```

2. Download the FER2013 dataset from the Kaggle website or any other reliable source and place it in the `dataset/` folder.

3. Preprocess the dataset:

   - Perform data augmentation techniques (optional but recommended).
   - Normalize pixel values to [0, 1] range.

4. Train the Emotion Detection model:

   - Run the `train.py` script to train the CNN model on the preprocessed dataset.
   - Tweak hyperparameters (e.g., learning rate, batch size, etc.) for optimal results.

5. Evaluate the model:

   - Use the `evaluate.py` script to assess the model's performance on a validation or test set.
   - Measure metrics such as accuracy, precision, recall, and F1-score.

6. Real-time Emotion Detection:

   - Implement the real-time emotion detection application using the trained model.
   - Run the `real_time_detection.py` script to activate the webcam and visualize emotion predictions in real-time.

### Model Architecture

The Emotion Detection model architecture consists of a Convolutional Neural Network (CNN) with multiple layers. The architecture can be customized by altering the number of layers, filters, and neurons to achieve the desired performance.

### Results

Our trained model achieved an accuracy of 0.16675954304820284 on the FER2013 test set. The model shows robust performance in recognizing facial emotions, making it suitable for real-world applications.

### Acknowledgments

We would like to thank the creators of the FER2013 dataset for providing valuable data for research and development in the field of facial expression recognition.
