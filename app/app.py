import streamlit as st
from routes.displacement import elastic_displacement
from routes.settlement import permanent_settlement
from routes.acceleration import acceleration_prediction, acceleration
import pickle
with open('models/v1/n_max_acc_rf_model.pkl', 'rb') as neg_file:
    n_max_rf_model = pickle.load(neg_file)

with open('models/v1/p_max_acc_rf_model.pkl', 'rb') as pos_file:
    p_max_rf_model = pickle.load(pos_file)

# Main App Configuration
st.set_page_config(page_title="Railway Track Prediction", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
nav_option = st.sidebar.radio("Choose Prediction Type:", ["Elastic Displacement", "Permanent Settlement", "Acceleration"])

# Header Tabs
tabs = st.tabs(["Predict", "Predict Multi-Aligned", "Train New Model"])

# Predict Section
with tabs[0]:
    st.title("Predict")
    # Dynamic content based on the sidebar selection
    if nav_option == "Elastic Displacement":
        st.subheader("Elastic Displacement Prediction")
        elastic_displacement()
    elif nav_option == "Permanent Settlement":
        st.subheader("Permanent Settlement Prediction")
        permanent_settlement()
    elif nav_option == "Acceleration":
        st.subheader("Acceleration Prediction")
        acceleration_prediction()


with tabs[1]:
    if nav_option == "Acceleration":
        st.title("Predict Multi-Aligned")
        acceleration()

# Train New Model Section
with tabs[2]:
    st.title("Train New Model")
    st.write("Upload your dataset to retrain the models.")

    # File upload for new training data
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        st.write("File uploaded successfully!")
        # Display uploaded file content
        import pandas as pd
        data = pd.read_csv(uploaded_file)
        st.dataframe(data)

        # Conditional training based on the sidebar selection
        if nav_option == "Elastic Displacement":
            st.info("Retraining Elastic Displacement Model")
            if st.button("Train Elastic Displacement Model"):
                st.write("Training Elastic Displacement Model... (Placeholder)")
        elif nav_option == "Permanent Settlement":
            st.info("Retraining Permanent Settlement Model")
            if st.button("Train Permanent Settlement Model"):
                st.write("Training Permanent Settlement Model... (Placeholder)")
