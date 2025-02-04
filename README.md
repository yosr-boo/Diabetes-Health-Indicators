# 📊 Diabetes Health Indicators - Machine Learning Project

## **Overview**

This project aims to develop machine learning models to predict whether an individual has **diabetes**, **prediabetes**, or is **healthy** based on data from the CDC’s Behavioral Risk Factor Surveillance System (BRFSS). By leveraging multiple supervised and unsupervised learning techniques, we compare model performances to identify the most effective approach for diabetes risk prediction.

## 🎯 **Objectives**
- Predict an individual's diabetes status using survey data.
- Identify key risk factors contributing to diabetes.
- Optimize models for healthcare applications, focusing on precision, recall, and F1-score.

## 🗂️ **Dataset**
- **Source:** CDC Diabetes Health Indicators Dataset (BRFSS 2015)  
- **Features:** 21 variables including **age**, **sex**, **BMI**, **blood pressure**, and **cholesterol** levels.  
- **Target Variable:** `Diabetes_binary` (0 = No Diabetes, 1 = Prediabetes/Diabetes)  
- **Special Characteristics:**
  - Balanced dataset (50-50 split for diabetic vs. non-diabetic cases).
  - Clean data with no missing values.
  - Pre-encoded target variable for binary classification.

## ⚙️ **Technologies & Tools**
- **Languages:** Python  
- **Libraries:** scikit-learn, pandas, NumPy, Matplotlib, Seaborn, Keras (for ANN)  
- **Techniques:**  
  - **Supervised Learning:** Logistic Regression, Decision Trees, Random Forest, Gradient Boosting Classifier, Neural Network (ANN), K-Nearest Neighbors (KNN)  
  - **Unsupervised Learning:** K-Means Clustering  
  - **Optimization:** Hyperparameter tuning, feature selection, normalization (Min-Max Scaling)

## 🚀 **Project Workflow**
1. **Data Preprocessing:**
   - Cleaning, balancing, and feature scaling using Min-Max Normalization.
   - Dropping irrelevant features based on correlation analysis.

2. **Exploratory Data Analysis (EDA):**
   - Visualizations (heatmaps, distribution plots) to understand feature relationships.
   - K-means clustering to identify natural data groupings.

3. **Model Development:**
   - **Supervised Learning Models:**  
     - Logistic Regression  
     - Decision Tree  
     - Random Forest  
     - Gradient Boosting Classifier  
     - Artificial Neural Network (ANN)  
     - K-Nearest Neighbors (KNN)
   - **Evaluation Metrics:**  
     - Accuracy, Precision, Recall, F1-Score, Confusion Matrix

4. **Model Optimization:**
   - Hyperparameter tuning for performance enhancement.
   - Cross-validation for robust evaluation.

5. **Feature Importance Analysis:**
   - Identifying the most predictive risk factors for diabetes, including **High Blood Pressure**, **High Cholesterol**, **General Health**, and **BMI**.

## 📊 **Key Results**
- **Best Model:** Gradient Boosting Classifier  
  - **Accuracy:** 75.5%  
  - **Recall:** 79.5% (best at identifying diabetic cases)  
  - **F1-Score:** 76%  
- **ANN Performance:** Strong recall (84.7%) but lower precision due to false positives.  
- **Logistic Regression & Random Forest:** Reliable baseline models with balanced performance.  
- **KNN:** Simple but less effective with high-dimensional data.

## 🧩 **Challenges & Limitations**
- **Class Imbalance:** Initially tackled through data balancing strategies.
- **Model Interpretability:** Complex models like ANN and Gradient Boosting are less interpretable compared to Decision Trees.
- **False Positives:** Trade-offs between recall and precision in healthcare applications.

## 🔑 **Insights & Conclusions**
- **Survey data can predict diabetes risk effectively**, especially with models like Gradient Boosting and ANN.
- **Top Risk Factors:** High Blood Pressure, High Cholesterol, General Health, Age, and BMI.
- A **short-form questionnaire** focusing on key features could maintain strong predictive performance.

## 📬 **Contact**
**Yosr Bouhoula**  
📧 [yosr_bouhoula@outlook.fr](mailto:yosr_bouhoula@outlook.fr)  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/yosr-bouhoula-5ab872151/)
