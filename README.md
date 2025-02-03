# IAM-Dataset-Maker
IAM Dataset Maker: Auto Annotation of Raw Handwritten Images A web application designed for cleaning and annotating raw handwritten text images, converting them into the IAM Format database for the model’s dataset.

# Documentation for Handwriting Recognition using CRNN

This document provides an overview of the code for a handwriting recognition system using a Convolutional Recurrent Neural Network (CRNN). The system is designed to recognize handwritten text from images using a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). The code also includes functionality to convert the dataset into the IAM format, which is commonly used for handwriting recognition tasks.

---

## Table of Contents
1. **Introduction**
2. **Code Overview**
   - 2.1 Custom Dataset (`HandwritingDataset`)
   - 2.2 Model Definition (`CRNN`)
   - 2.3 Training Loop (`train_model`)
   - 2.4 Dataset and DataLoader
   - 2.5 Model, Loss, and Optimizer
   - 2.6 Conversion to IAM Format (`convert_to_iam_format`)
3. **Dependencies**
4. **Usage**
5. **Limitations and Future Work**

---

## 1. Introduction
The code implements a handwriting recognition system using a CRNN model. The system processes images of handwritten text, extracts features using a CNN, and then uses an RNN to predict the text. The model is trained using the Connectionist Temporal Classification (CTC) loss, which is suitable for sequence prediction tasks like handwriting recognition.

The code also includes functionality to convert the dataset into the IAM format, which is a standard format for handwriting recognition datasets.

---

## 2. Code Overview

### 2.1 Custom Dataset (`HandwritingDataset`)
The `HandwritingDataset` class is a custom PyTorch dataset that loads images from a specified directory and extracts labels using Optical Character Recognition (OCR) via Tesseract.

- **Attributes:**
  - `image_dir`: Directory containing the images.
  - `image_files`: List of image filenames in the directory.
  - `transform`: Transformations to apply to the images (e.g., resizing, converting to tensor).

- **Methods:**
  - `__len__()`: Returns the number of images in the dataset.
  - `__getitem__(idx)`: Loads and returns the image and its corresponding label (extracted using OCR).
  - `get_label(image)`: Uses Tesseract to extract text from the image.

### 2.2 Model Definition (`CRNN`)
The `CRNN` class defines the Convolutional Recurrent Neural Network model.

- **Attributes:**
  - `cnn`: A CNN module that extracts features from the input images.
  - `rnn`: An RNN module (LSTM) that processes the sequence of features.
  - `fc`: A fully connected layer that maps the RNN output to the final character predictions.

- **Methods:**
  - `forward(x)`: Defines the forward pass of the model.

### 2.3 Training Loop (`train_model`)
The `train_model` function trains the CRNN model using the provided DataLoader.

- **Parameters:**
  - `model`: The CRNN model to train.
  - `dataloader`: DataLoader providing the training data.
  - `criterion`: Loss function (CTC loss).
  - `optimizer`: Optimizer (Adam).
  - `num_epochs`: Number of training epochs.

- **Process:**
  - Iterates over the dataset for the specified number of epochs.
  - Computes the loss, performs backpropagation, and updates the model weights.

### 2.4 Dataset and DataLoader
The dataset and DataLoader are created using the `HandwritingDataset` class.

- **Parameters:**
  - `image_dir`: Directory containing the images.
  - `transform`: Transformations to apply to the images.

- **DataLoader:**
  - `batch_size`: Number of images per batch.
  - `shuffle`: Whether to shuffle the data.

### 2.5 Model, Loss, and Optimizer
The model, loss function, and optimizer are initialized here.

- **Model:**
  - `CRNN(imgH=32, nc=1, nclass=37, nh=256)`: Initializes the CRNN model with specified parameters.
    - `imgH`: Height of the input images.
    - `nc`: Number of input channels (1 for grayscale).
    - `nclass`: Number of output classes (37 for alphanumeric characters).
    - `nh`: Number of hidden units in the RNN.

- **Loss Function:**
  - `nn.CTCLoss()`: CTC loss for sequence prediction.

- **Optimizer:**
  - `optim.Adam(model.parameters(), lr=0.001)`: Adam optimizer with a learning rate of 0.001.

### 2.6 Conversion to IAM Format (`convert_to_iam_format`)
The `convert_to_iam_format` function converts the dataset into the IAM format.

- **Parameters:**
  - `image_dir`: Directory containing the images.
  - `model`: Trained CRNN model.
  - `transform`: Transformations to apply to the images.

- **Process:**
  - Creates a directory for the IAM dataset.
  - Processes each image, predicts the text using the model, and saves the prediction in IAM format.

- **Helper Function:**
  - `decode_output(output)`: Decodes the model's output into readable text (to be implemented).

---

## 3. Dependencies
The code requires the following Python libraries:
- `torch`
- `torchvision`
- `PIL` (Pillow)
- `pytesseract`
- `numpy`

Install the dependencies using:
```bash
pip install torch torchvision pillow pytesseract numpy
```

---

## 4. Usage
1. Place your handwritten images in a directory (e.g., `./images`).
2. Update the `image_dir` variable with the path to your image directory.
3. Run the script to train the model and convert the dataset to IAM format.

---

## 5. Limitations and Future Work
- **OCR Accuracy:** The current implementation uses Tesseract for label extraction, which may not be accurate for all handwriting styles.
- **Decoding Output:** The `decode_output` function needs to be implemented to decode the model's output into text.
- **Dataset Size:** The model's performance depends on the size and quality of the training dataset.
- **IAM Format:** The IAM format conversion is a basic implementation and may need adjustments for specific use cases.

---

This documentation provides a high-level overview of the code. For detailed implementation, refer to the code comments and PyTorch documentation.