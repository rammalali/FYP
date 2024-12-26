import streamlit as st
from routes.displacement import Displacement
from routes.settlement import Settlement
from routes.acceleration import Acceleration
from routes.train import Train
import pandas as pd
import pickle
import tempfile


acceleration = Acceleration()
displacement = Displacement()
settlement = Settlement()


# CSS for cell styling
cell_style = """
    <style>
    .cell-container {
        background-color: #e6f7ff;
        border: 1px solid #91d5ff;
        padding: 15px;
        margin: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    </style>
    """

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
        displacement.elastic_displacement()
    elif nav_option == "Permanent Settlement":
        st.subheader("Permanent Settlement Prediction")
        settlement.permanent_settlement()
    elif nav_option == "Acceleration":
        st.subheader("Acceleration Prediction")
        acceleration.acceleration_prediction()


with tabs[1]:
    if nav_option == "Elastic Displacement":
        st.title("Predict Multi-Aligned")
        displacement.displacement_multi_aligned(cell_style)
    elif nav_option == "Permanent Settlement":
        st.title("Predict Multi-Aligned")
        settlement.settlement_multi_aligned(cell_style)
    if nav_option == "Acceleration":
        st.title("Predict Multi-Aligned")
        acceleration.acceleration_multi_aligned(cell_style)

# Train New Model Section
with tabs[2]:
    training_complete = False
    st.title("Train New Model")
    st.write("Upload your dataset to retrain the models.")

    # Initialize session state variables
    if "trained_model" not in st.session_state:
        st.session_state.trained_model = None
    if "training_complete" not in st.session_state:
        st.session_state.training_complete = False

    # Train New Model Section
    st.title("Train New Model")
    st.write("Upload your dataset to retrain the models.")

    # File upload for new training data
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        st.write("File uploaded successfully!")
        st.write(nav_option)
        # Determine the file type and load the data
        try:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                st.stop()

            st.dataframe(data)

            # Initialize Train class
            train_instance = Train(data, nav_option)

            if train_instance.check_df_data():
                st.success(f"Data validation passed for {nav_option} Model. Ready to train.")

                # Conditional training based on the sidebar selection
                if st.button(f"Train {nav_option} Model"):
                    with st.spinner(f"Training {nav_option} Model. Please wait..."):
                        trained_model, test_score = train_instance.train_model()
                        st.session_state.trained_model = trained_model
                        st.session_state.training_complete = True
                    st.success(f"{nav_option} Model training completed.")

                    # Display test score in two columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Model Training Completed**")
                    with col2:
                        st.metric(label="Test Score", value=f"{test_score:.4f}")

            else:
                st.error("The uploaded data does not meet the requirements for the selected model type.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

    # Display save button only if training is complete
    if st.session_state.training_complete and st.session_state.trained_model:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
            pickle.dump(st.session_state.trained_model, tmp_file)
            tmp_file_path = tmp_file.name

        with open(tmp_file_path, "rb") as file:
            st.download_button(
                label="Download Trained Model",
                data=file,
                file_name=f"{nav_option.replace(' ', '_').lower()}_trained_model.pkl",
                mime="application/octet-stream"
            )
