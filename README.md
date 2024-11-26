# Lab 2: Neural Networks for Computer Vision with PyTorch

## Objective

The main objective of this lab is to become familiar with the PyTorch library and to build various neural architectures, including Convolutional Neural Networks (CNN), Faster R-CNN, Fully Connected Neural Networks (FCNN), and Vision Transformers (ViT) for computer vision tasks.

## Work to Do

### Part 1: CNN Classifier

#### Dataset

- The MNIST dataset consists of grayscale images of handwritten digits (0–9) with dimensions **28x28 pixels**.
- **MNIST Dataset:** [MNIST Dataset on Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

1. **CNN Architecture:**
   - Build a CNN model using PyTorch for classifying the MNIST dataset.
   - Define layers (Convolution, Pooling, Fully Connected).
   - Set hyperparameters (e.g., kernels, padding, stride, optimizers, regularization, etc.).
   - Train the model using GPU.

  2. **Faster R-CNN:**
   - Implement a Faster R-CNN architecture for classifying the MNIST dataset.

3. **Comparison of CNN and Faster R-CNN:**
   - Compare the performance of CNN and Faster R-CNN using various metrics, including:
     - Accuracy
     - F1 Score
     - Loss
     - Training time

  ## Metrics for Comparison

| Metric           | CNN                          | Faster R-CNN                  |
|-------------------|------------------------------|--------------------------------|
| **Accuracy**      | To be filled after training  | To be filled after training    |
| **F1 Score**      | To be filled after training  | To be filled after training    |
| **Loss**          | To be filled after training  | To be filled after training    |
| **Training Time** | To be filled after training  | To be filled after training    |

---

4. **Fine-Tuning with Pretrained Models:**
   - Fine-tune the CNN and Faster R-CNN models using pretrained models like VGG16 and AlexNet on the MNIST dataset.
   - Compare the results of the fine-tuned models with the original CNN and Faster R-CNN models.
   - Discuss the conclusions based on the comparison.
  

### Part 2: Vision Transformer (ViT)

#### Vision Transformers (ViT)

Since their introduction by Dosovitskiy et al. in 2020, Vision Transformers (ViT) have dominated the field of computer vision, achieving state-of-the-art performance in image classification and other tasks.

#### Tasks:

1. **Build Vision Transformer Model:**
   - Follow the tutorial on Medium: [Vision Transformers from Scratch in PyTorch](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c) to establish a ViT model from scratch.
   - Apply the model for the classification task on the MNIST dataset.

2. **Interpret and Compare Results:**
