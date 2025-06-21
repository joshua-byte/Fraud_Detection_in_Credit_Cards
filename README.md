# 🕵️ Credit Card Fraud Detection – Machine Learning Project

A complete fraud detection pipeline using logistic regression on the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). This project handles class imbalance, applies ML modeling, and outputs predictions and insights.

---

## 📁 Dataset
- Source: Kaggle – [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions with only 492 fraud cases (highly imbalanced)

---

## 🔧 Tools & Technologies
- Python, Pandas, NumPy, scikit-learn, Plotly
- SMOTE (imbalanced-learn) for class balancing
- Logistic Regression as baseline classifier
- Excel export for business usability

---

## ⚙️ Workflow Summary

### 1. Data Preprocessing
- Checked nulls, duplicates, and info structure
- Scaled numeric features using `StandardScaler`
- Dropped `Time` and separated `Class` as label

### 2. Resampling with SMOTE
- Applied SMOTE to balance fraud vs. non-fraud ratio
- Ensured proper stratified train-test split

### 3. Model Training
- Trained a Logistic Regression model
- Evaluated with classification report and accuracy score

### 4. Feature Importance
- Extracted feature weights from the logistic regression
- Visualized with Plotly bar chart

### 5. Prediction & Export
- Generated predictions on test set
- Exported output as `fraud_predictions.xlsx`
- Visualized `Actual vs. Predicted` using scatter plot

---

## 📊 Results

**Model Accuracy:** ~95% (on balanced data with SMOTE)  
---

## 📸 Screenshots
![feature_importance](https://github.com/user-attachments/assets/2ddea064-83d6-4420-8351-e37debbc8a93)
![Actual_vs_Predictions](https://github.com/user-attachments/assets/fb259a79-7daf-4cbd-a202-18a4a2ad37c3)


## 🔚 Conclusion

This project demonstrates a complete fraud detection pipeline on real-world financial data — from preprocessing and balancing to modeling, visualization, and export-ready outputs.

---

## 📬 Contact

Built by [@joshua-byte](https://github.com/joshua-byte)  
