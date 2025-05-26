import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import time

st.set_page_config(page_title="Streamlit All-in-One", layout="wide")

# Title
st.title("🧠 All-in-One Streamlit Dashboard")

# Sidebar navigation
menu = st.sidebar.radio("Navigation", ["🏠 Home", "📁 Upload Data", "📈 Visualize", "🤖 ML Model"])

# 1. HOME
if menu == "🏠 Home":
    st.header("Welcome to the Streamlit Showcase App")
    st.markdown("""
    This app demonstrates **major features** of Streamlit:
    - Widgets
    - File upload
    - Data visualization
    - ML prediction
    - Session state
    - Layouts and caching
    """)
    st.image("https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png", width=300)

# 2. UPLOAD DATA
elif menu == "📁 Upload Data":
    st.header("Upload Your CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file:
        @st.cache_data
        def load_data(file):
            return pd.read_csv(file)

        df = load_data(uploaded_file)
        st.success("✅ Data Loaded Successfully!")
        st.dataframe(df.head())

        # Show data info
        if st.checkbox("Show summary statistics"):
            st.write(df.describe())

# 3. VISUALIZE
elif menu == "📈 Visualize":
    st.header("Visualize Data with Seaborn and Plotly")
    uploaded_file = st.file_uploader("Upload CSV", key="viz", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        st.subheader("Choose Columns")
        x_col = st.selectbox("X-axis", numeric_cols)
        y_col = st.selectbox("Y-axis", numeric_cols, index=1)
        chart_type = st.radio("Chart Type", ["Scatter", "Line", "Bar", "Plotly Scatter"])

        st.subheader("Generated Chart")
        if chart_type == "Scatter":
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            st.pyplot(fig)
        elif chart_type == "Line":
            fig, ax = plt.subplots()
            sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
            st.pyplot(fig)
        elif chart_type == "Bar":
            fig, ax = plt.subplots()
            sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
            st.pyplot(fig)
        elif chart_type == "Plotly Scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title="Interactive Plotly Chart")
            st.plotly_chart(fig)

# 4. ML MODEL
elif menu == "🤖 ML Model":
    st.header("Simple ML Predictor")

    st.markdown("Use sample features to test a pre-trained model.")
    col1, col2 = st.columns(2)
    with col1:
        a = st.slider("Feature 1", 0.0, 10.0)
        b = st.slider("Feature 2", 0.0, 10.0)
    with col2:
        c = st.slider("Feature 3", 0.0, 10.0)
        d = st.slider("Feature 4", 0.0, 10.0)

    # Simulated model
    st.subheader("Prediction Result")
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    fake_model_result = a + b + c + d
    st.success(f"✅ Predicted Value: {round(fake_model_result, 2)}")

    if fake_model_result > 20:
        st.error("Warning: High risk!")
    else:
        st.info("Normal range.")

