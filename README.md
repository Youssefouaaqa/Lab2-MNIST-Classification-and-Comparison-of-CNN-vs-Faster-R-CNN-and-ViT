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

| Metric            | CNN                          | Faster R-CNN                   |
|-------------------|------------------------------|--------------------------------|
| **Accuracy**      | 99.24%                       | To be filled after training    |
| **F1 Score**      | 0.9924                       | To be filled after training    |
| **Loss**          | 0.0198                       | To be filled after training    |
| **Training Time** | 141.10 seconds               | To be filled after training    |

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
   **Key Insights**:
   - Accuracy: CNN achieves the highest accuracy (99.24%), significantly outperforming both Faster R-CNN and ViT (76.07%).
   - F1 Score: The CNN model also has a high F1 score (0.9924), indicating strong precision and recall. No F1 score data is available for Faster R-CNN and ViT.
   - Loss: CNN has a much lower loss (0.0198) compared to Faster R-CNN (1.543) and ViT (1.70), suggesting better performance in terms of minimizing error during training.
   - Training Time: CNN is the fastest to train, taking only 141.10 seconds. Faster R-CNN, being more complex due to its object detection capabilities, takes over an hour, while ViT has a relatively faster training time of 30 minutes.
   - 
In summary, CNN excels in accuracy, loss, and training time, while Faster R-CNN is more complex with longer training times, and ViT offers faster training but at a cost of lower accuracy.
