# 📊 Churn Prediction — Predicting Customer Attrition with ANN
**Python Jupyter Notebook**

---

## 🧠 Overview
This project focuses on predicting **customer churn** using **Artificial Neural Networks (ANN)**. Churn occurs when customers stop doing business with a company. Predicting churn is crucial in sectors like **banking** and **subscription-based services**, where retaining customers is more profitable than acquiring new ones.

The project demonstrates:

- ✅ Data cleaning & preprocessing  
- 📊 Exploratory Data Analysis (EDA) with visualizations  
- 🎨 Feature transformation and encoding  
- 🤖 Building and training an ANN for binary classification  
- 📈 Model evaluation (accuracy, training/validation curves)  

---

## 📂 Dataset
- **Source:** `Churn_Modelling.csv`  
- **Number of records:** 10,000  
- **Features include:**  

| Feature | Description |
|---------|-------------|
| CreditScore | Customer credit score |
| Geography | Country (France, Germany, Spain) |
| Gender | Customer gender |
| Age | Customer age |
| Tenure | Number of years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products held |
| HasCrCard | Credit card ownership (0/1) |
| IsActiveMember | Active customer flag (0/1) |
| EstimatedSalary | Estimated annual salary |
| Exited | Target variable (0 = retained, 1 = churned) |

**Key Stats:**  
- Age: Min 18, Max 92, Mean 38.9  
- Balance: Mean 76,486 €  
- Churn prevalence: 20%  

---

## 🎯 Objectives
1. Perform **Exploratory Data Analysis** to uncover patterns  
2. Preprocess data: encode categorical variables, scale numerical features  
3. Build a **feedforward neural network** for churn prediction  
4. Evaluate model performance with accuracy and loss metrics  
5. Analyze key drivers of churn  

---

## 💡 Domain Knowledge
- Older customers tend to churn more  
- Inactive members have higher attrition  
- Customers with fewer products are more likely to leave  
- Geography may influence churn  

---

## 🧰 Tools & Libraries
**Python Installations**
```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
```

**Python Imports**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

---

## ⚙️ How to Run
1. Clone the repository  
```bash
git clone https://github.com/AlexandraB12/churn-prediction-ann.git
```
2. Navigate into the project folder  
```bash
cd churn-prediction-ann
```
3. Launch the Jupyter Notebook  
```bash
jupyter notebook main.ipynb
```

---

## 📈 Exploratory Data Analysis
- **Target distribution:** 20% churned, 80% retained  
- **Insights:**  
  - Older and inactive customers have higher churn  
  - Fewer products → higher churn  
  - High account balances correlate with higher churn  

---

## 🔧 Preprocessing & Feature Engineering
- Drop irrelevant columns: `RowNumber`, `CustomerId`, `Surname`  
- Encode categorical variables: Geography & Gender  
- Standardize numerical features  
- Convert target variable `Exited` to categorical  

---

## 🤖 Modeling
- Feedforward ANN:  
  - Input → 10 neurons, ReLU  
  - Hidden → 7 neurons, ReLU + Dropout + BatchNorm  
  - Output → 2 neurons, Sigmoid  
- Optimizer: Adam  
- Loss: Binary Crossentropy  
- Epochs: 50  
- Train/Validation split: 80/20  

---

## 📊 Evaluation
- Accuracy: 86%  
- Main drivers of churn: Age, Tenure, Balance, Activity status  
- High-balance inactive customers at risk  

---

## 📌 Key Takeaways
- ANN predicts churn effectively (86% accuracy)  
- Age, Balance, Tenure, Activity status are top predictors  
- Provides actionable insights for retention strategies  

---

## 🔮 Next Steps
- Test deeper ANN architectures  
- Compare with Random Forest or XGBoost  
- Use SHAP values for interpretability  
- Collect additional features (interactions, complaints, support data)  
- Explore SMOTE or class weighting for imbalance  

---

## 🧾 Author
**Alexandra Boudia**  
Data Scientist | Machine Learning Enthusiast | Predictive Analytics  
🔗 Connect on LinkedIn


