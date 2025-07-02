# Image Classification on CIFAR-10 using Custom CNN and AlexNet

## 1. Project Overview

This project implements and evaluates two Convolutional Neural Network (CNN) architectures for image classification on the CIFAR-10 dataset. The primary goal is to compare the performance of a simple, custom-built CNN against a deeper, AlexNet-inspired model. Additionally, the project explores the effect of targeted data augmentation on improving the accuracy of poorly performing classes.

---

## 2. Key Objectives

* **Design and Implement a Custom CNN:** Create a lightweight CNN from scratch to establish a performance baseline.
* **Implement an AlexNet-inspired Architecture:** Adapt the classic AlexNet model for the 32x32 image size of CIFAR-10.
* **Compare Model Performance:** Evaluate and contrast the accuracy, training time, and complexity (number of parameters) of the two models.
* **Improve Weak Class Predictions:** Use data augmentation as a strategy to boost the accuracy of classes that the models initially struggle with.

---

## 3. Models Implemented

### a. Custom CNN

A simple and efficient CNN with the following architecture:

* **Conv2D Layer:** 32 filters, 3x3 kernel, ReLU activation, followed by Batch Normalization.
* **MaxPooling2D Layer:** 2x2 pool size.
* **Conv2D Layer:** 64 filters, 3x3 kernel, ReLU activation, followed by Batch Normalization.
* **MaxPooling2D Layer:** 2x2 pool size.
* **Flatten** Layer
* **Dropout Layer:** Rate of 0.25 for regularization.
* **Dense Layer:** 128 units, ReLU activation.
* **Dense Output Layer:** 10 units (for 10 classes), Softmax activation.

### b. AlexNet-Inspired Model

A deeper and more complex architecture based on the original AlexNet, adapted for CIFAR-10:

* **Five Convolutional Layers:** With filter sizes of 96, 256, 384, 384, and 256, all using ReLU activation and Batch Normalization.
* **Three MaxPooling Layers:** Interspersed between the convolutional layers.
* **Two Fully Connected Dense Layers:** Each with 1024 units and a Dropout rate of 0.5 for regularization.
* **A Softmax Output Layer.**

---

## 4. Methodology and Experiments

1.  **Initial Training:** Both the Custom CNN and the AlexNet model were trained for 15 epochs on the CIFAR-10 training dataset.
2.  **Performance Analysis:** After the initial training, the models were evaluated on the test set. Per-class accuracy was analyzed using confusion matrices, which revealed that both models struggled with classes like "cat" and "dog."
3.  **Data Augmentation:** To address this, a targeted data augmentation strategy was employed. New training images were generated for the four weakest classes ('bird', 'cat', 'deer', 'dog') using:
    * Rotation (±15 degrees)
    * Horizontal Flips
    * Zooming (0.9-1.1 scale)
4.  **Re-training:** The augmented images were added to the original training set, and both models were re-trained on this expanded dataset to evaluate the impact of the augmentation.

---

## 5. Results and Key Findings

The final test accuracies highlight the trade-offs between the models and the unexpected impact of the augmentation strategy.

| Model                       | Test Accuracy (Before Augmentation) | Test Accuracy (After Augmentation) |
| --------------------------- | ----------------------------------- | ---------------------------------- |
| Custom CNN                  | 59.2%                               | 55.5%                              |
| AlexNet-inspired Model      | **75.1%** | 72.3%                              |

### Key Findings:

* **AlexNet Outperforms Custom CNN:** The deeper AlexNet architecture achieved significantly higher accuracy than the simpler custom model, demonstrating the benefit of a more complex model for this task.
* **Targeted Augmentation Decreased Accuracy:** Contrary to expectations, adding augmented data for only the weak classes *lowered* the overall test accuracy for both models. This is likely due to **class imbalance**, where the models became biased towards the over-represented augmented classes, hurting their ability to generalize across the entire balanced test set.

---

## 6. Future Improvements

### Practical Next Steps

* **Augment All Data Equally:** Apply simple augmentations (like random flips and rotations) to *all* training images. This increases dataset diversity without causing the class imbalance that previously hindered performance.
* **Tune the Training Process:** Experiment with different learning rates, train for more epochs, and adjust dropout values to find a better configuration for the AlexNet model.
* **Advanced Optimization:** Use a **learning rate scheduler** (like Cosine Annealing) to dynamically adjust the learning rate during training and add **weight decay (L2 regularization)** to better control the training process.
