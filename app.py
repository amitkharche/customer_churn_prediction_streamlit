
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv("churn_data.csv")

st.title("Customer Churn Prediction Dashboard")
st.write("This dashboard demonstrates a basic Random Forest model on synthetic churn data.")

# Show data
if st.checkbox("Show Raw Data"):
    st.write(df.head())

# Split data
X = df.drop("churn", axis=1)
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# Classification report
st.subheader("Classification Report")
st.dataframe(pd.DataFrame(report).transpose())

# Confusion matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Feature importance
st.subheader("Feature Importance")
feat_importance = pd.Series(model.feature_importances_, index=X.columns)
st.bar_chart(feat_importance.sort_values(ascending=False))
