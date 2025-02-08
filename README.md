# Image-Classification
# Image Classification using CNN

## Overview
This project implements an image classification model using Convolutional Neural Networks (CNNs) with TensorFlow and Keras. The model is trained on the CIFAR-10 dataset, which consists of 60,000 colored images categorized into 10 different classes.

## Motivation
The project aims to leverage CNNs' capabilities in automating image classification tasks, which are crucial in applications like facial recognition, medical diagnosis, and autonomous vehicles.

## Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

## Dataset
The CIFAR-10 dataset contains:
- **50,000 training images**
- **10,000 test images**
- **10 Classes:** Plane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck

## Model Architecture
The CNN model consists of:
1. **Convolutional Layers:** Extract features from images using filters.
2. **Pooling Layers:** Reduce image dimensions while preserving important features.
3. **Flattening Layer:** Converts feature maps into a vector format.
4. **Fully Connected (Dense) Layers:** Classifies the images into respective categories.
5. **Activation Functions:** ReLU for hidden layers and SoftMax for classification.

## Implementation Steps
1. **Import Libraries:** NumPy, TensorFlow, Matplotlib.
2. **Load Dataset:** Use CIFAR-10 dataset available in Keras.
3. **Preprocessing:** Normalize pixel values between 0 and 1.
4. **Build CNN Model:**
   - Convolution and MaxPooling layers.
   - Flattening and Dense layers with ReLU and SoftMax activation.
5. **Compile Model:** Using Adam optimizer and categorical cross-entropy loss function.
6. **Train the Model:** Train with multiple epochs to improve accuracy.
7. **Evaluate Performance:** Measure accuracy and loss on the test dataset.

## Observations
- Initial ANN model had low accuracy (~50%).
- Switching to CNN improved accuracy (~80%).
- Model performed well with clear images but struggled with noisy inputs.

## Challenges
- Dataset quality and size impact accuracy.
- Training requires high computational power.

## Conclusion
This project demonstrates the effectiveness of CNNs in image classification tasks. It lays the foundation for more advanced applications like object detection and multi-class classification.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/image-classification.git
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib
   ```
3. Run the script:
   ```bash
   python train_model.py
   ```

## Future Enhancements
- Implement data augmentation for better generalization.
- Use a deeper CNN model for improved accuracy.
- Experiment with different optimization techniques.
