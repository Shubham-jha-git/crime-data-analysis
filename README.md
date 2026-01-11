# Crime Data Analysis and Prediction

## 📌 Project Overview
This project is a **Crime Data Analysis and Prediction System** developed using Python and Machine Learning.  
It analyzes crime patterns across different Indian states and predicts future crime counts based on user-selected inputs.

The project also includes an **interactive web dashboard** built using **Streamlit** for real-time prediction and visualization.

---

## 📊 Features
- State-wise and crime-type-wise analysis
- Year-wise crime trend visualization
- Comparison of different crime categories
- Future crime prediction using a Machine Learning model
- Interactive web interface using Streamlit

---

## 🧠 Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

---

## 🤖 Machine Learning Model
- Model Used: **Linear Regression**
- Type: Supervised Learning
- Output: Predicted Crime Count (Numerical)

The trained model is saved and reused for predictions without retraining.

---

## 🌐 Web Application (Streamlit)
The project includes a Streamlit-based web application that allows users to:
- Select State
- Select Crime Type (or All Crimes Combined)
- Select Year (including future years)
- View predicted crime count
- View crime trend and comparison graphs

---

## 📁 Dataset Information
- The dataset used in this project is **synthetically generated** using Python and Pandas.
- It simulates year-wise crime data for educational and analytical purposes.
- No real or government crime data is used.

---

## 📂 Project Structure


##  Project Structure
crime-data-analysis/
│
├── app/
│ └── app.py # Streamlit web application
│
├── data/
│ ├── raw/
│ └── processed/
│
├── models/
│ └── crime_prediction_model.pkl
│
├── notebooks/
│ └── Jupyter notebooks for analysis and model training
│
├── requirements.txt
└── README.md


---

## ▶️ How to Run the Project

### 1️⃣ Activate Environment
```bash
conda activate crime_env
streamlit run app/app.py

---

##  Exploratory Data Analysis
- Year-wise crime trend analysis
- State-wise crime comparison
- Crime-type distribution analysis

---

##  Machine Learning Model
- Algorithm Used: Linear Regression
- Target Variable: Crime_Count
- Evaluation Metrics: R² Score, Mean Absolute Error (MAE)

Note: Since the dataset is synthetically generated, model accuracy is not the primary focus.

---

🎓 Academic Note
This project is developed as a college mini-project to demonstrate:
Data analysis
Machine learning workflow
Visualization
Deployment using Streamlit

✅ Conclusion

The project successfully demonstrates an end-to-end data analytics and machine learning pipeline with an interactive web-based interface.
---

## 👤 Author
Shubham Jha  
BSc IT (Second Year)

