# 🌍 Language Detection using Recurrent Neural Network (RNN)

This project implements a **Language Detection System** using a **Simple Recurrent Neural Network (RNN)** and deploys it as an interactive **Streamlit web application**.  
Given an input sentence, the model predicts the **language of the text** along with a confidence score.

## 🚀 Project Overview

- Built a sequence classification model using **Keras SimpleRNN**
- Trained on tokenized and padded text sequences
- Deployed using **Streamlit** for real-time predictions
- Uses a saved trained model and tokenizer for inference

The application allows users to enter any sentence and instantly detect its language.

## 🧠 Model Architecture

- Embedding Layer  
- Simple RNN Layer  
- Dense Output Layer with Softmax  
The model is trained to perform **multi-class classification** over supported languages.

## 🖥️ Web Application

The Streamlit app:
- Takes user input text  
- Converts it into sequences using a saved tokenizer  
- Pads the sequence to fixed length  
- Predicts language using the trained RNN model  
- Displays:
  - Predicted Language  
  - Confidence Score  



## 📁 Project Structure

├── app.py # Streamlit application
├── saved_model/
│ ├── simple_rnn_model.h5 # Trained RNN model
│ └── tokenizer.pkl # Tokenizer + Label Encoder
├── requirements.txt
├── notebooks/
│ └── training.ipynb # Model training notebook
└── README.md


---

## ⚙️ Tech Stack
**Programming Language**
- Python 3.10+

**Libraries & Frameworks**
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Streamlit  
- Pickle  

**Tools**
- VS Code  
- Git & GitHub  

📊 Example Output
Input:
यह एक अच्छा दिन है
Output:
Predicted Language: Hindi
Confidence: 0.97

🧩 Key Features
End-to-end NLP pipeline
Real-time inference
Clean and simple UI
Reusable trained artifacts
Suitable for deployment demos

📌 Future Improvements
Add LSTM / GRU comparison
Support more languages
Add confusion matrix and evaluation metrics
Deploy on cloud (Streamlit Cloud / HuggingFace Spaces)

👨‍💻 Author:
Kumar Ketan |PreFinal Year CS Student|Focus: Machine Learning, Deep Learning, DSA
🔗 LinkedIn:https://www.linkedin.com/in/kumar-ketan-5456b531b/
🔗 GitHub:https://github.com/Kumar-Ketan




