import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============ PAGE CONFIG ============
st.set_page_config(page_title="Employee Attrition Prediction Dashboard", layout="wide")

# ============ HEADER ============
st.title("Employee Turnover Prediction Dashboard")
st.markdown("""
This interactive dashboard showcases insights, analysis, and prediction results from the **Employee Attrition Prediction** project.
Upload your dataset or explore the visual analytics below.
""")

# ============ SIDEBAR ============
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to Section", ["Overview", "Exploratory Data Analysis", "Model Results", "Decision Tree", "Predict Attrition"])

# ============ DATA LOADING ============
@st.cache_data
def load_data():
    df = pd.read_excel("Attrition.xlsx")
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

@st.cache_data
def train_model(df):
    df_clean = df.copy()
    le_dict = {}
    for col in df_clean.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        le_dict[col] = le

    X = df_clean.drop("Attrition", axis=1)
    y = df_clean["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model, X, y, le_dict, X_test, y_test

# Train model once and reuse
model, X, y, le_dict, X_test, y_test = train_model(df)

# ============ SECTION 1: OVERVIEW ============
if section == "Overview":
    st.subheader("Dataset Overview")
    st.write("### Data Preview")
    st.dataframe(df.head())

    st.write("### Summary Statistics")
    st.dataframe(df.describe(include="all"))

    st.write("### Missing Values")
    st.dataframe(df.isnull().sum())



# ============ SECTION 2: EDA ============
elif section == "Exploratory Data Analysis":
    st.subheader("Exploratory Data Analysis")

    # Attrition distribution
    st.markdown("### Attrition Distribution")
    fig1 = px.pie(df, names="Attrition", title="Attrition Ratio", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig1, use_container_width=True)

    # Separate numeric & categorical columns
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # ---------- Categorical vs Attrition ----------
    st.markdown("### Categorical Features vs Attrition")
    for col in cat_cols:
        if col != "Attrition":
            fig = px.histogram(df, x=col, color="Attrition", barmode="group",
                               title=f"{col} vs Attrition", color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

    # ---------- Numerical vs Attrition ----------
    st.markdown("### Numerical Features vs Attrition")
    for col in num_cols:
        if col != "Attrition":
            fig = px.box(df, x="Attrition", y=col, color="Attrition",
                         title=f"{col} vs Attrition", color_discrete_sequence=px.colors.qualitative.Prism)
            st.plotly_chart(fig, use_container_width=True)

    # ---------- Correlation Heatmap ----------
    st.markdown("### Correlation Heatmap (Numeric Features Only)")
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 8))  # larger size for better visibility
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.3, cbar_kws={'shrink': 0.7}, ax=ax)
    plt.title("Correlation Heatmap", fontsize=16, fontweight='bold', pad=20)
    st.pyplot(fig, clear_figure=True)


# ============ SECTION 3: MODEL RESULTS ============
elif section == "Model Results":
    st.subheader(" Model Training and Evaluation")

    st.write("Training a **Decision Tree Classifier** on the dataset...")

    df_clean = df.copy()
    le = LabelEncoder()
    for col in df_clean.select_dtypes(include=["object"]).columns:
        df_clean[col] = le.fit_transform(df_clean[col])

    X = df_clean.drop("Attrition", axis=1)
    y = df_clean["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{acc*100:.2f}%")
    col2.metric("Test Samples", len(y_test))

    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    st.pyplot(fig)

    st.write("### Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

# ============ SECTION 4: DECISION TREE ============s
elif section == "Decision Tree":
    st.subheader("Decision Tree Visualization")
    st.markdown("Below is the visualization of the trained Decision Tree used for prediction.")

    import os
    tree_path = os.path.join(os.path.dirname(__file__), "decision_tree.png")

    if os.path.exists(tree_path):
        # Display image with full width using 'width' parameter
        st.image(tree_path, caption="Decision Tree Model", width=900)
    else:
        st.warning(" decision_tree.png not found — please ensure it's in the same directory as app.py")

# ============ SECTION 5: PREDICTION FORM ============
elif section == "Predict Attrition":
    st.subheader(" Predict Employee Attrition")
    st.markdown("Enter employee details below to predict whether they are likely to leave the organization:")

    sample_features = {}
    for col in df.columns:
        if col == "Attrition":
            continue
        if df[col].dtype == "object":
            sample_features[col] = st.selectbox(f"{col}", df[col].unique())
        else:
            sample_features[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    if st.button("Predict"):
        input_df = pd.DataFrame([sample_features])
        input_df_encoded = input_df.copy()
        for col in input_df_encoded.select_dtypes(include=["object"]).columns:
            if col in le_dict:
                input_df_encoded[col] = le_dict[col].transform(input_df_encoded[col])

        pred = model.predict(input_df_encoded)[0]
        if pred == 1:
            st.error("Employee is likely to leave (Attrition = Yes)")
        else:
            st.success("Employee is likely to stay (Attrition = No)")
