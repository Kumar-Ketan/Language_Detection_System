# 🌍 Language Detection Using RNN

## 🔄 Workflow
Data Collection  
↓  
Data Preprocessing  
↓  
Text Tokenization & Padding  
↓  
RNN Model Training  
↓  
Model Evaluation  
↓  
Model Saving  
↓  
Model Loading  
↓  
Streamlit Web Application  

---

## 📌 Project Overview
This project focuses on **automatic language detection** from text using a  
**Recurrent Neural Network (RNN)**.

The model is trained on a **multilingual language detection dataset** and deployed using  
**Streamlit** to provide **real-time predictions and interactive visualization**.

The application is divided into **three major components**:
- **Model Training**
- **Model Prediction**
- **Web App Deployment using Streamlit**

---

## 🧠 Problem Statement
Language detection is a fundamental task in **Natural Language Processing (NLP)**.

Accurate identification of language helps in:
- Text classification systems
- Multilingual applications
- Translation systems
- Content moderation

---

## ⚙️ Project Structure
```
LANGUAGE_DETECTION/
│
├── saved_model/
│   ├── simple_rnn_model.h5
│   └── tokenizer.pkl
│
├── datasets/
│   └── Language Detection.csv
│
├── app.py
├── eda.ipynb
├── prediction.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧩 Project Components

### 1️⃣ Model Training
- Dataset used: `Language Detection.csv`
- Text preprocessing:
  - Text cleaning
  - Tokenization
  - Sequence padding
- RNN model built using **TensorFlow / Keras**
- Model trained to learn language patterns
- Trained artifacts saved:
  - `simple_rnn_model.h5`
  - `tokenizer.pkl`

---

### 2️⃣ Model Prediction
- Loads trained RNN model and tokenizer
- Accepts raw text input
- Predicts the **language of the given text**
- Prediction workflow demonstrated in:
  - `prediction.ipynb`

---

### 3️⃣ Exploratory Data Analysis (EDA)
- Language distribution analysis
- Dataset insights and visualizations
- Implemented in:
  - `eda.ipynb`

---

### 4️⃣ Streamlit Web Application
- Interactive user interface
- Real-time language detection
- Clean and intuitive design
- Main application file:
  - `app.py`

---

## 🛠️ Tech Stack

### 👨‍💻 Programming Language
- Python

### 📚 Libraries & Frameworks
- NumPy  
- Pandas  
- TensorFlow / Keras  
- Pickle  

### 🌐 Deployment & Visualization
- Streamlit

---

## 📊 Dataset
- **Language Detection Dataset**
- Multilingual text data

---

## 🚀 Installation & Execution
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Author
**Kumar Ketan**  
Python |SQL| AI & ML|  

---

## ⭐ Acknowledgement
Thanks to open-source datasets and libraries that made this project possible.

---

## 🚀 Connect With Me
📧 Email: kketan6205@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/kumar-ketan-5456b531b/  
 🐙 GitHub:  https://github.com/Kumar-Ketan  


---

⭐ Thanks for checking out this project!
