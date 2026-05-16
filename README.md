#  Traffic Sign Recognition using CNN (GTSRB)

This project implements a Convolutional Neural Network (CNN) for classifying traffic signs using the **GTSRB dataset**. It includes model training, evaluation, and an interactive **Streamlit web application**.

---

##  Project Overview

The goal of this project is to classify German traffic signs into **43 categories** using deep learning.

This project demonstrates:
- Image preprocessing and augmentation
- CNN model training
- Model evaluation and visualization
- Real-time inference using Streamlit

---

##  Dataset

- **Name:** GTSRB (German Traffic Sign Recognition Benchmark)
- **Classes:** 43
- **Image size:** 32x32 (resized)
- **Type:** Multi-class image classification

---

##  Model Architecture

The CNN model includes:

- Convolutional layers (Conv2D)
- Batch Normalization
- MaxPooling layers
- Dropout (regularization)
- Fully connected Dense layers
- Softmax activation (43 classes)

---

##  Results

- Training Accuracy: ~99%
- Validation Accuracy: ~99%
- Strong generalization on unseen test images

---

##  Streamlit Web App

The app allows users to:

- Upload a traffic sign image
- Get real-time prediction
- View confidence score
- See top-5 predicted classes

---

##  How to Run

Run Streamlit app

streamlit run app.py

Project Structure:

├── app.py                     # Streamlit web application

├── traffic_sign_model.keras  # Trained CNN model

├── notebook.ipynb            # Model training notebook

├── README.md                 # Project documentation

---

## 🧪 Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

---

##  Evaluation Strategy

Model performance was evaluated using:

- Training vs Validation accuracy curves
- Loss curves analysis
- Random test image prediction
- Class-wise prediction analysis (Top-5 probabilities)

---

##  Key Insights

- Model converges quickly after few epochs
- High accuracy (~99%) on validation set
- Strong generalization on random unseen images
- No strong signs of overfitting observed

---

##  Sample Output

- Upload image → Prediction → Confidence score
- Top-5 class probability visualization

---

## Future Improvements

- Grad-CAM explainability (visual attention maps)
- Real-time webcam detection
- Model optimization for edge devices
- Cloud deployment (Streamlit Cloud / HuggingFace Spaces)

---

## Author

Ali Rafati

---

## Support

If you like this project, please consider giving it a ⭐ on GitHub!
