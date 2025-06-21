# INSOMNIA_PREDICTION
Insomnia Analysis and Prediction using Machine Learning Techniques

# DESCRIPTION 
This project presents a real-time web application that predicts insomnia risk levels among university students aged 19–25 using machine learning techniques. It is an accessible, non-invasive, and cost-effective alternative to traditional clinical methods like polysomnography.

## 📌 Project Overview

College students often face irregular sleep patterns due to academic stress, excessive screen exposure, and poor sleep hygiene. This project addresses the need for early detection of insomnia by leveraging AI for risk classification into:

- ✅ No Risk
- ⚠️ At Risk
- 🔴 High Risk

## 🎯 Objectives

- Predict insomnia risk using behavioral and lifestyle attributes.
- Provide real-time assessment through a user-friendly web app.
- Compare model performance using Random Forest and XGBoost.


## 📂 Project Structure

📁 Insomnia-Risk-Predictor/
├── app.py                  # Streamlit app
├── preprocess.py          # Data preprocessing functions
├── insomnia\_model.pkl     # Trained Random Forest model
├── insomnia\_xgb\_model.pkl # Trained XGBoost model
├── insomnia\_encoder.pkl   # Label encoder for categorical variables
├── feature\_columns.json   # Feature schema for model input
├── cleaned\_insomnia\_data.csv # Final cleaned dataset
└── README.md              # Project documentation


## 🧠 Technologies Used

- **Language:** Python 3.9+
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Streamlit, Joblib, Seaborn, Matplotlib
- **IDE/Tools:** Jupyter Notebook, VS Code, Google Forms (data collection)
- **Deployment:** Localhost using Streamlit


## 📊 Dataset Overview

- **Source:** Google Forms survey among university students (Age 19–25)
- **Total Records:** 600
- **Features:** Sleep hours, screen time, caffeine intake, stress levels, etc.
- **Target:** Sleep Risk Category (No Risk, At Risk, High Risk)

## 🏗️ Methodology

1. **Data Collection** – Structured Google Form responses.
2. **Preprocessing** – Label encoding, handling missing data, standardization.
3. **Modeling** – Trained and compared multiple classifiers.
4. **Evaluation** – Accuracy, precision, recall, F1-score, confusion matrix.
5. **Deployment** – Real-time prediction using Streamlit.

## 🚀 How to Run the App

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/insomnia-risk-predictor.git
   cd insomnia-risk-predictor

2. Install required packages:

   bash
   pip install -r requirements.txt

3. Launch the Streamlit app:

   bash
   streamlit run app.py

4. Enter your lifestyle data in the form and choose a prediction model to get your result instantly.


## ✅ Features

* Dual-model selection: Random Forest & XGBoost
* Streamlit-based responsive UI
* Real-time risk prediction
* Clean and preprocessed dataset for reproducibility
* Accuracy: **Random Forest – 93%**, **XGBoost – 94%**


## 📈 Model Performance

| Metric       | Random Forest | XGBoost |
| ------------ | ------------- | ------- |
| Accuracy     | 93%           | 94%     |
| Macro F1     | 0.93          | 0.94    |
| Real-time UI | ✅             | ✅       |


## 🔮 Future Enhancements

* Add personalized feedback or lifestyle recommendations
* Support wearable device integration (Fitbit, Oura Ring, etc.)
* Mobile app version
* Multilingual UI for wider accessibility
* Add deep learning-based hybrid models

##
Here’s a clear and proper **GitHub README instruction** (step-by-step) for your project (`insomnia_rf_streamlit_app.zip`) explaining which file to run first and the procedure to execute the project properly.



# 📝How to Run the Insomnia Prediction Streamlit App

### **Step 1: Clone the Repository**

bash
git clone https://github.com/yourusername/insomnia_rf_streamlit_app.git
cd insomnia_rf_streamlit_app


### **Step 2: Create and Activate Virtual Environment (Optional but Recommended)**

#### On **Windows**:

bash
python -m venv venv
venv\Scripts\activate


#### On **macOS/Linux**:

bash
python3 -m venv venv
source venv/bin/activate


### **Step 3: Install Required Packages**

bash
pip install -r requirements.txt


If there is no `requirements.txt`, you can manually install:

bash
pip install pandas scikit-learn openpyxl streamlit joblib


### **Step 4: Train the Model (Run this first)**

You **must train the model first** to create the model file before running the app.

bash
python train_model.py


✔️ This script will:

* Read the dataset (`Balanced_Risk_Levels.xlsx`)
* Preprocess the data
* Train the RandomForest model
* Save the trained model as a `.pkl` file (usually `random_forest_model.pkl`)


### **Step 5: Run the Streamlit Web App**

After the model is trained successfully, launch the Streamlit app by:

bash
streamlit run app.py


✔️ This will open a browser window where you can interact with the insomnia risk prediction model.


### **Important Files in the Project**

| File Name                       | Purpose                                                                |
| ------------------------------- | ---------------------------------------------------------------------- |
| **train\_model.py**             | Trains the model and saves it as a pickle file. Must be run first.     |
| **app.py**                      | Streamlit app that loads the trained model and performs predictions.   |
| **preprocess.py**               | Contains functions to preprocess the input data.                       |
| **Balanced\_Risk\_Levels.xlsx** | Input dataset used for model training.                                 |
| **random\_forest\_model.pkl**   | (Generated after running `train_model.py`) Pickled trained model file. |
| **requirements.txt**            | (Optional) Contains all required Python packages.                      |



### **Step 6: (Optional) Push the Project to GitHub**

bash
git add .
git commit -m "Initial commit with Insomnia Prediction Streamlit App"
git push origin main


## ⚠️ **Important Notes**:

* **Do not run `app.py` before training** the model using `train_model.py`, else the app will fail because the model file (`random_forest_model.pkl`) won't exist.
* Ensure that the file paths inside `train_model.py` and `app.py` are correct relative to the project structure.



