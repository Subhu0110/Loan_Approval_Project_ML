# 🏦 Loan Approval Prediction using Traditional & Ensemble Machine Learning Models

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-purple)

## 📌 Project Overview

Loan approval is a critical decision-making process in the banking and financial sector. Incorrect loan approvals can lead to financial losses, while rejecting eligible applicants may result in missed business opportunities.

This project develops a machine learning-based loan approval prediction system that analyzes applicant information and predicts whether a loan application should be **Approved** or **Rejected**.

The project also compares multiple classification algorithms, including traditional machine learning models and ensemble learning techniques, to identify the most effective approach for loan approval prediction.

---

# 🎯 Objectives

* Predict loan approval status using applicant financial information.
* Compare the performance of traditional and ensemble machine learning models.
* Analyze how different algorithms behave on the same dataset.
* Identify the best-performing model for the task.

---

# 📂 Dataset Information

The dataset contains applicant financial and demographic information.

### Features Used

| Feature                  | Description              |
| ------------------------ | ------------------------ |
| no_of_dependents         | Number of dependents     |
| education                | Education level          |
| self_employed            | Employment status        |
| income_annum             | Annual income            |
| loan_amount              | Requested loan amount    |
| loan_term                | Loan duration            |
| cibil_score              | Credit score             |
| residential_assets_value | Residential assets value |
| commercial_assets_value  | Commercial assets value  |
| luxury_assets_value      | Luxury assets value      |
| bank_asset_value         | Bank assets value        |
| loan_status              | Target Variable          |

### Target Variable

| Value | Meaning  |
| ----- | -------- |
| 1     | Approved |
| 0     | Rejected |

---

# 🔄 Project Workflow

## 1️⃣ Data Loading

The dataset was imported using Pandas and inspected for structure, data types, and missing values.

```python
df = pd.read_csv("loan_approval_dataset.csv")
```

---

## 2️⃣ Data Cleaning

Several preprocessing steps were performed:

* Removed leading spaces from column names.
* Removed leading spaces from categorical values.
* Removed the `loan_id` column as it acts only as an identifier.
* Converted the target variable into binary numerical format.

---

## 3️⃣ Exploratory Data Analysis (EDA)

EDA was performed to understand:

* Loan approval distribution
* Income patterns
* Asset value distributions
* Relationships between numerical variables

### Visualizations Used

✅ Class Distribution Plots

✅ Histograms

✅ Correlation Heatmap

---

## 4️⃣ Feature Engineering

Categorical variables were converted into numerical representations using:

```python
pd.get_dummies()
```

This allowed machine learning algorithms to process the data effectively.

---

## 5️⃣ Train-Test Split

The dataset was divided into training and testing sets.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

---

## 6️⃣ Feature Scaling

Standard Scaling was applied to algorithms sensitive to feature magnitudes.

### Models Scaled

* Logistic Regression
* KNN
* SVM
* Naive Bayes

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Tree-based models were trained using unscaled data.

---

# 🤖 Machine Learning Models Implemented

## Traditional Machine Learning Models

### Logistic Regression

A linear classification algorithm used as the baseline model.

### K-Nearest Neighbors (KNN)

Predicts classes based on nearest neighboring observations.

### Support Vector Machine (SVM)

Finds the optimal decision boundary separating classes.

### Gaussian Naive Bayes

Probabilistic classifier based on Bayes Theorem.

---

## Tree-Based Models

### Decision Tree

Creates decision rules through recursive splitting of data.

---

## Ensemble Learning Models

### Random Forest (Bagging)

Combines multiple decision trees to improve predictive performance and reduce overfitting.

### AdaBoost (Boosting)

Sequentially trains weak learners by focusing on previously misclassified samples.

### Stacking Classifier

Combines predictions from multiple base models using a meta-model for final prediction.

---

# 📊 Model Performance

| Model                | Accuracy   |
| -------------------- | ---------- |
| Logistic Regression  | 90.51%     |
| KNN                  | 89.22%     |
| SVM                  | 92.38%     |
| Naive Bayes          | 93.67%     |
| Decision Tree        | 97.42%     |
| AdaBoost             | 96.72%     |
| Stacking Classifier  | 97.30%     |
| **Random Forest** 🏆 | **97.77%** |

---

# 📈 Performance Analysis

### Top 3 Models

🥇 Random Forest — 97.77%

🥈 Decision Tree — 97.42%

🥉 Stacking Classifier — 97.30%

### Key Observation

Ensemble learning methods consistently outperformed traditional machine learning models.

The Random Forest Classifier achieved the highest accuracy and demonstrated excellent generalization performance.

---

# 💡 Key Findings

✅ Random Forest achieved the highest accuracy of **97.77%**.

✅ Ensemble learning methods outperformed traditional machine learning models.

✅ Decision Tree achieved 97.42% accuracy, indicating strong decision-rule patterns within the dataset.

✅ Stacking Classifier performed exceptionally well but did not surpass Random Forest.

✅ AdaBoost achieved 96.72% accuracy, confirming the effectiveness of boosting techniques.

✅ Naive Bayes outperformed Logistic Regression, KNN, and SVM despite being a simpler probabilistic model.

✅ Financial indicators such as income, assets, and credit score play a significant role in determining loan approval decisions.

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Scikit-Learn

---

# 📁 Repository Structure

```text
Loan-Approval-Prediction-Using-Ensemble-Learning
│
├── Loan_Approval_Prediction.ipynb
├── loan_approval_dataset.csv
├── README.md
├── requirements.txt
│
└── images
    ├── model_comparison.png
    └── correlation_heatmap.png
```

---

# 🚀 Future Improvements

* ROC-AUC Analysis
* Hyperparameter Tuning
* Cross Validation
* XGBoost Implementation
* Feature Importance Analysis
* Streamlit Deployment

---

# 👨‍💻 Author

## Subhansh Yadav

**B.Tech CSE (AI & ML)**

IIIT Nagpur

Machine Learning • Data Science • Artificial Intelligence

---

⭐ If you found this project interesting, consider giving the repository a star.
