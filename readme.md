# Language Detection using RNN

## 📌 Project Description

This project is a **Language Detection System** built using a Recurrent Neural Network (RNN) model.  
It takes an input sentence and predicts the language of the text with a confidence score.  
The model is deployed using a **Streamlit web application** for easy interaction.

This project demonstrates the complete machine learning workflow:
- Data preprocessing  
- Model training  
- Model saving/loading  
- Web-based inference using Streamlit  

## 🚀 Features

- Detects language from user input text  
- Displays predicted language and confidence percentage  
- Simple and clean Streamlit UI  
- Uses trained RNN model for prediction  

## 🛠️ Tech Stack

- **Programming Language:** Python 3.11  
- **Deep Learning:** TensorFlow / Keras  
- **Data Processing:** NumPy, Pandas  
- **Web Framework:** Streamlit  
- **Model Storage:** HDF5 (`.h5`), Pickle  

## 📂 Project Structure
language_detection/
│
├── app.py # Streamlit application
├── simple_rnn_model.h5 # Trained RNN model
├── tokenizer.pkl # Saved tokenizer
├── requirements.txt # Project dependencies
├── eda.ipynb # Data analysis notebook
├── prediction.ipynb # Testing notebook
├── datasets/
│ └── Language Detection.csv
└── README.md


---

## ⚙️ How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone <https://github.com/Kumar-Ketan/Language_Detection_System>
cd language_detection
2. Create & Activate Conda Environment (Python 3.11)
conda create -n venv311 python=3.11
conda activate venv311

3. Install Dependencies
pip install -r requirements.txt

4. Run the Streamlit App
streamlit run app.py

🧪 Model Details
Model Type: Recurrent Neural Network (RNN)
Input: Raw text sentence
Output: Predicted language + confidence score
Trained on multilingual text dataset

📈 Future Improvements
Add more languages
Improve accuracy using LSTM/GRU
Add batch prediction support
Deploy on cloud (Heroku / Render / HuggingFace Spaces)

## 👤 Author

**Kumar Ketan**

Pre-Final Year CS Student  
Focus: Machine Learning, Deep Learning, DSA  

- 🔗 LinkedIn: https://www.linkedin.com/in/kumar-ketan-5456b531b/  
- 💻 GitHub: https://github.com/Kumar-Ketan  





