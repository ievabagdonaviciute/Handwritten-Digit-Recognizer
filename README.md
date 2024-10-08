# Handwritten-Digit-Recognizer

This repository contains code for a **Handwritten Digit Recognition** project using machine learning techniques. The goal is to classify digits (0-9) based on image inputs from the MNIST dataset.

## Project Overview

The project uses a neural network model to recognize handwritten digits from images. The model is trained using the MNIST dataset and can predict the correct digit from new images.

### Key Files
- **`main.py`**: The main script for training the neural network model using the MNIST dataset. This file includes model architecture setup, data loading, training, and evaluation.
- **`testing.py`**: A script for testing the model's accuracy on new input images. This file contains code to load the trained model and make predictions on unseen data.

## Dependencies

To run this project, you will need the following Python libraries:
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV (for handling image inputs)

You can install these libraries using:
```bash
pip install tensorflow keras numpy matplotlib opencv-python
```

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/ievabagdonaviciute/Handwritten-Digit-Recognizer.git
   ```

2. Navigate to the project folder:
   ```bash
   cd Handwritten-Digit-Recognizer
   ```

3. Train the model:
   ```bash
   python main.py
   ```

4. Test the model with new input:
   ```bash
   python testing.py
   ```

## Dataset

The project uses the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for training and testing the model. The dataset consists of 60,000 training images and 10,000 testing images of handwritten digits.

## Results

After training, the model achieves an accuracy of approximately X% on the test set (update with your results).

## Future Work

- Improve model accuracy by experimenting with different neural network architectures.
- Add data augmentation techniques to increase the dataset size and improve model robustness.
- Implement a graphical user interface (GUI) to allow users to upload images for real-time digit recognition.


## Credits

The code for this project was inspired by the YouTube tutorial "[Handwritten Digit Recognition using Neural Networks](https://www.youtube.com/watch?v=Zi4i7Q0zrBs)" by **NeuralNine**. Special thanks for the detailed walkthrough!

