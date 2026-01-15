# Customer Churn Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview

This project focuses on predicting whether a customer will **stay with the company or leave (churn)** using an **Artificial Neural Network (ANN)**. The model is trained on a churn dataset and deployed using **Streamlit** for interactive visualization and real-time predictions.

The complete solution is divided into three major components:

1. **Model Training** – Data preprocessing, ANN model building, and training.
2. **Model Prediction** – Using the trained model to predict customer churn.
3. **Visualization & Deployment** – Streamlit-based web app for user interaction.

---

## 🚀 Features

* End-to-end churn prediction pipeline
* Data preprocessing with scaling and encoding
* ANN-based classification model
* Real-time predictions
* Interactive Streamlit dashboard
* Easy-to-use interface
* Model persistence and reuse

---

## 🧠 Model Description

The ANN model is designed with multiple dense layers using the ReLU activation function and a sigmoid output layer for binary classification.

* Input Layer: Customer features
* Hidden Layers: Dense layers with ReLU activation
* Output Layer: Sigmoid activation (Churn / No Churn)

---

## 🛠 Tech Stack

### Programming Language

* Python

### Libraries & Frameworks

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras
* Pickle
* Streamlit

### Tools

* Jupyter Notebook
* VS Code
* Git & GitHub

---

## 📂 Project Structure

```
Churn_Prediction/
│
├── dataset/
│   └── Churn_Modelling.csv
│
├── Saved_model/
│   ├── model.h5 / model.keras
│   └── preprocessor.pkl
│
├── notebooks/
│   ├── eda.ipynb
│   ├── training.ipynb
│   └── prediction.ipynb
│
├── app.py        # Streamlit app
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

1. Clone the repository

```bash
git clone <repository_url>
cd Churn_Prediction
```

2. Create a virtual environment

```bash
python -m venv venv
```

3. Activate the virtual environment

```bash
venv\Scripts\activate   # Windows
```

4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Model Training

* Data is cleaned and preprocessed.
* Categorical features are encoded.
* Numerical features are scaled.
* ANN model is trained using binary cross-entropy loss and Adam optimizer.
* Model performance is evaluated using accuracy and validation loss.

---

## 🔍 Model Prediction

The trained model is used to predict whether a customer will:

* **Stay with the company** ✅
* **Leave the company (Churn)** ❌

Predictions are made using new customer input data after preprocessing.

---

## 🌐 Streamlit Deployment

The Streamlit app provides:

* User input fields for customer details
* Real-time churn prediction
* Interactive UI
* Clean and simple interface

To run the app:

```bash
streamlit run app.py
```

---

## 📊 Results

* Achieved high accuracy on validation data
* Stable training with minimal overfitting
* Reliable churn prediction

---

## 👤 Author

**Name:** Sanchi Preet Kaur
**Email:** [spk99110@gmail.com](mailto:spk99110@gmail.com)
**Role:** Data Science & Machine Learning Enthusiast
**Project Type:** Academic / Learning Project

---

## 🚀 Connect With Me

📧 Email: [spk99110@gmail.com](mailto:spk99110@gmail.com)
🐙 GitHub: (sanchi-preet-kaur)
🔗 LinkedIn: [https://www.linkedin.com/in/sanchi-preet-kaur-0443b12a4](https://www.linkedin.com/in/sanchi-preet-kaur-0443b12a4)

---

## ⭐ Acknowledgement

Thanks to open-source datasets, libraries, and the developer community that made this project possible.

---

## 📜 License

This project is for educational purposes.

---

## ⭐ Acknowledgements

* Kaggle / Public Churn Dataset
* TensorFlow Documentation
* Streamlit Community

---

Feel free to contribute, suggest improvements, or raise issues!
