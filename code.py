# %%
import pandas as pd
import plotly.express as px
import numpy as np
import lightgbm as lgb

df = pd.read_csv("creditcard.csv")
df

# %%
df.info()

# %%
df.isnull().sum()

# %%
df.duplicated()

# %%
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
scaling = [col for col in df.columns if col not in ["Class", "Time"]]
df[scaling] = scale.fit_transform(df[scaling])

# %%
X = df.drop(columns=["Class", "Time"])
y = df["Class"]

# %%
fraudcount = df["Class"].value_counts()
fp = (fraudcount[1] / fraudcount.sum()) * 100
fp

# %%
from imblearn.over_sampling import SMOTE
from collections import Counter

X_resample = df.drop(columns=["Class", "Time"])  # Exclude target and time columns
y_resample = df["Class"]


smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_resample, y_resample)

# Check new class distribution
new_class_distribution = Counter(y_smote)
new_class_distribution


# %%
from sklearn.linear_model import LogisticRegression

#Feature Importance
log_reg = LogisticRegression(max_iter=500, random_state=42)
log_reg.fit(X, y)

feature_importances_logreg = np.abs(log_reg.coef_[0])

importance_df_logreg = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances_logreg})
importance_df_logreg = importance_df_logreg.sort_values(by="Importance", ascending=False)


fig = px.bar(
    importance_df_logreg.head(10),  # Top 10 important features
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 10 Most Important Features for Fraud Detection (Logistic Regression)",
    labels={"Importance": "Feature Importance Score", "Feature": "Feature"},
    color="Importance",
    color_continuous_scale="viridis"
)

fig.show()


# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42, stratify=y_smote)


# %%
# Train the model
model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train, y_train)

# %%
# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(report)

# %%
results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(results)

# %%
df_plot = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})


df_plot["Actual"] += np.random.normal(0, 0.02, df_plot.shape[0])
df_plot["Predicted"] += np.random.normal(0, 0.02, df_plot.shape[0])


fig = px.scatter(df_plot, x="Actual", y="Predicted", color="Actual",
                 title="Actual vs. Predicted Fraud Cases",
                 labels={"Actual": "Actual Fraud Label", "Predicted": "Predicted Fraud Label"},
                 opacity=0.6)

fig.show()

# %%
# Ensure df and y_pred have matching lengths
df_test = df.iloc[:len(y_pred)].copy()  
df_test['Predicted Fraud'] = y_pred

# Save as Excel (fixed typo)
df_test.to_excel('fraud_predictions.xlsx', index=False)



