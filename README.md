# How-PCA-dimensionality-reduction-affects-MNIST-digit-classification-accuracy

## **Overview**
This notebook analyses how Principal Component Analysis (PCA) dimensionality reduction affects MNIST digit classification accuracy.

---

## **Objectives**
- Load the MNIST dataset  
- Define and compile a Keras classification model  
- Evaluate model performance on PCA-reduced data **without** retraining  
- Retrain the model on reduced representations and re-evaluate  
- Visualise test accuracy vs. number of PCA components

---

## **Methodology**
- **Setup:**  
  - Import TensorFlow/Keras and scikit-learn’s PCA  
  - Confirm TensorFlow version  

- **Model Definition:**  
  - `create_model()` returns a Sequential network (Dense → ReLU → Softmax)  

- **Evaluation Functions:**  
  - `evaluate_pca_accuracy(N)`: project test data onto top N components  
  - `plot_pca_accuracy(range_N)`: plot accuracy curve  

- **Experiments:**  
  - Test accuracy on unreduced vs. PCA-projected data  
  - Retrain model on PCA data for various N  
  - Identify optimal N for minimal accuracy loss

---

## **Key Results**
- Accuracy peaks around:  
  - **Original model:** N = 784 components  
  - **Retrained model:** N ≈ 300 components  
- Beyond mid-range N, accuracy plateaus—most discriminatory features captured early  
- Demonstrates MNIST compressibility via linear PCA

---

## **Limitations & Future Work**
- Limited to linear PCA; consider kernel PCA or autoencoders  
- Validate on additional image datasets (e.g., CIFAR-10)  
- Explore trade-offs between dimensionality and inference speed

