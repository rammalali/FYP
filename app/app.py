import streamlit as st
from routes.displacement import Displacement
from routes.settlement import Settlement
from routes.acceleration import Acceleration
from routes.train import Train
import pandas as pd
import pickle
import tempfile

with open('models/v1/n_max_acc_rf_model.pkl', 'rb') as neg_file:
    n_max_rf_model = pickle.load(neg_file)
with open('models/v1/p_max_acc_rf_model.pkl', 'rb') as pos_file:
    p_max_rf_model = pickle.load(pos_file)

with open('models/v1/ps_rf_model.pkl', 'rb') as file:
    ps_loaded_model = pickle.load(file)

with open('models/v1/smoothed_displacement_rf_model.pkl', 'rb') as file:
    sd_loaded_model = pickle.load(file)


acceleration = Acceleration(n_max_rf_model=n_max_rf_model, p_max_rf_model=p_max_rf_model)
displacement = Displacement(loaded_model=sd_loaded_model)
settlement = Settlement(loaded_model=ps_loaded_model)


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

def check_model_features(model, feature_names):
    """
    Check if the model can accept the required features.
    """
    try:
        # Create a dummy dataframe with the required features
        test_df = pd.DataFrame({feature: [0] for feature in feature_names})
        model.predict(test_df)
        return True
    except Exception:
        return False
    
def upload_model(nav_option):
    # Initialize state for multi-aligned prediction
    if nav_option == "Acceleration":
        st.session_state.is_acceleration = True
    else:
        st.session_state.is_acceleration = False

    # Display file upload options based on the state
    if st.session_state.is_acceleration:
        uploaded_file_n = st.file_uploader("Upload Negative Acceleration Model (.pkl):", type=["pkl"], key="model_n")
        uploaded_file_p = st.file_uploader("Upload Positive Acceleration Model (.pkl):", type=["pkl"], key="model_p")

        if uploaded_file_n and uploaded_file_p:
            try:
                pretrained_model_n = pickle.load(uploaded_file_n)
                pretrained_model_p = pickle.load(uploaded_file_p)

                if not check_model_features(pretrained_model_n, ['velocity', 'Cycle_Number']) or \
                   not check_model_features(pretrained_model_p, ['velocity', 'Cycle_Number']):
                    st.error("One or both models do not support the required features: 'velocity' and 'Cycle_Number'.")
                else:
                    st.success("Both models uploaded successfully and verified!")
                    acceleration.set_n_max_rf_model(pretrained_model_n)
                    acceleration.set_p_max_rf_model(pretrained_model_p)
                    acceleration.acceleration_multi_aligned(cell_style)
            except Exception as e:
                st.error(f"Failed to load the models. Error: {e}")
    else:
        uploaded_file = st.file_uploader("Upload your pretrained model (.pkl):", type=["pkl"], key="model")

        if uploaded_file:
            try:
                pretrained_model = pickle.load(uploaded_file)
                if not check_model_features(pretrained_model, ['velocity', 'Cycle_Number']):
                    st.error("The uploaded model does not support the required features: 'velocity' and 'Cycle_Number'.")
                else:
                    st.success("Model uploaded successfully and verified!")
                    # Pass the model to the appropriate handler
            except Exception as e:
                st.error(f"Failed to load the model. Error: {e}")

# Predict Section
with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        st.title("Predict")

    pretrained_model = None
    pretrained_model_n = None
    pretrained_model_p = None

    with col2:
        if nav_option == "Acceleration":
            st.session_state.is_acceleration = True
        else:
            st.session_state.is_acceleration = False

        # Display file upload options based on the state
        if st.session_state.is_acceleration:
            uploaded_file_n = st.file_uploader("Upload Negative Acceleration Model (.pkl):", type=["pkl"], key="model_n")
            uploaded_file_p = st.file_uploader("Upload Positive Acceleration Model (.pkl):", type=["pkl"], key="model_p")

            if uploaded_file_n and uploaded_file_p:
                try:
                    pretrained_model_n = pickle.load(uploaded_file_n)
                    pretrained_model_p = pickle.load(uploaded_file_p)

                    if not check_model_features(pretrained_model_n, ['velocity', 'Cycle_Number']) or \
                    not check_model_features(pretrained_model_p, ['velocity', 'Cycle_Number']):
                        st.error("One or both models do not support the required features: 'velocity' and 'Cycle_Number'.")
                    else:
                        st.success("Both models uploaded successfully and verified!")

                except Exception as e:
                    st.error(f"Failed to load the models. Error: {e}")
        else:
            uploaded_file = st.file_uploader("Upload your pretrained model (.pkl):", type=["pkl"])
            if uploaded_file:
                try:
                    pretrained_model = pickle.load(uploaded_file)
                    if not check_model_features(pretrained_model, ['velocity', 'Cycle_Number']):
                        st.error("The uploaded model does not support the required features: 'velocity' and 'Cycle_Number'.")
                        pretrained_model = None
                    else:
                        st.success("Model uploaded successfully and verified!")
                except Exception as e:
                    st.error(f"Failed to load the model. Error: {e}")
                    pretrained_model = None

    with col1:
        if nav_option == "Elastic Displacement":
            st.subheader("Elastic Displacement")
            print(pretrained_model)
            model = pretrained_model if pretrained_model else displacement.loaded_model
            displacement.set_loaded_model(model)
            plot_image, res_table = displacement.elastic_displacement()
            if plot_image:
                with col1:
                    st.subheader("Predicted Displacement Table")
                    st.dataframe(res_table)
                with col2:
                    st.subheader("Predicted Displacement Graph")
                    st.image(plot_image, caption="Predicted Displacement", use_container_width=True)

        elif nav_option == "Permanent Settlement":
            st.subheader("Permanent Settlement")
            model = pretrained_model if pretrained_model else settlement.loaded_model
            settlement.set_loaded_model(model)
            plot_image, res_table = settlement.permanent_settlement()
            if plot_image:
                with col1:
                    st.subheader("Predicted Settlement Table")
                    st.dataframe(res_table)
                with col2:
                    st.subheader("Predicted Settlement Graph")
                    st.image(plot_image, caption="Predicted Settlement", use_container_width=True)

        elif nav_option == "Acceleration":
            st.subheader("Acceleration")
            print(pretrained_model_n, pretrained_model_p)

            n_model = pretrained_model_n if pretrained_model_n else acceleration.n_max_rf_model
            p_model = pretrained_model_p if pretrained_model_p else acceleration.p_max_rf_model
            acceleration.set_n_max_rf_model(n_model)
            acceleration.set_p_max_rf_model(p_model)

            plot_image = acceleration.acceleration_prediction()
            if plot_image:

                with col2:
                    st.subheader("Acceleration Predictions")
                    st.image(plot_image, caption="Acceleration Prediction", use_container_width=True)

st.session_state.is_acceleration = False
with tabs[1]:
    left, right = st.columns(2)
    st.write("Upload models and perform predictions for multiple aligned data.")

    # Initialize state for multi-aligned prediction
    if nav_option == "Acceleration":
        st.session_state.is_acceleration = True
    else:
        st.session_state.is_acceleration = False

    # Display file upload options based on the state
    if st.session_state.is_acceleration:
        uploaded_file_n = st.file_uploader("Upload Negative Acceleration Model (.pkl):", type=["pkl"], key="multi_n")
        uploaded_file_p = st.file_uploader("Upload Positive Acceleration Model (.pkl):", type=["pkl"], key="multi_p")

        if uploaded_file_n and uploaded_file_p:
            try:
                pretrained_model_n = pickle.load(uploaded_file_n)
                pretrained_model_p = pickle.load(uploaded_file_p)

                if not check_model_features(pretrained_model_n, ['velocity', 'Cycle_Number']) or \
                   not check_model_features(pretrained_model_p, ['velocity', 'Cycle_Number']):
                    st.error("One or both models do not support the required features: 'velocity' and 'Cycle_Number'.")
                else:
                    st.success("Both models uploaded successfully and verified!")

            except Exception as e:
                st.error(f"Failed to load the models. Error: {e}")
    else:
        uploaded_file = st.file_uploader("Upload your pretrained model (.pkl):", type=["pkl"], key="multi")

        if uploaded_file:
            try:
                pretrained_model = pickle.load(uploaded_file)
                if not check_model_features(pretrained_model, ['velocity', 'Cycle_Number']):
                    st.error("The uploaded model does not support the required features: 'velocity' and 'Cycle_Number'.")
                else:
                    st.success("Model uploaded successfully and verified!")
                    # Pass the model to the appropriate handler
            except Exception as e:
                st.error(f"Failed to load the model. Error: {e}")

    with left:
        st.title("Predict Multi-Aligned")



    if nav_option == "Elastic Displacement":
        model = pretrained_model if pretrained_model else displacement.loaded_model
        displacement.set_loaded_model(model)
        displacement.displacement_multi_aligned(cell_style)
    elif nav_option == "Permanent Settlement":
        model = pretrained_model if pretrained_model else settlement.loaded_model
        settlement.set_loaded_model(model)
        settlement.settlement_multi_aligned(cell_style)
    if nav_option == "Acceleration":
        n_model = pretrained_model_n if pretrained_model_n else acceleration.n_max_rf_model
        p_model = pretrained_model_p if pretrained_model_p else acceleration.p_max_rf_model
        acceleration.set_n_max_rf_model(n_model)
        acceleration.set_p_max_rf_model(p_model)

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

    left, right = st.columns(2)
    train_instance = None

    with right:
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

                # Initialize Train class
                train_instance = Train(data, nav_option)
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")

    with left:

        if train_instance:
            st.dataframe(data)


    col1, col2 = st.columns(2)
    if train_instance:
        if train_instance.check_df_data():
            st.success(f"Data validation passed for {nav_option} Model. Ready to train.")
    
        if train_instance.check_df_data():
            # # Conditional training based on the sidebar selection
            # train_btn = st.button(f"Train {nav_option} Model")
            with col1:
                train_btn = st.button(f"Train {nav_option} Model")
                if train_btn:
                    with st.spinner(f"Training {nav_option} Model. Please wait..."):
                        trained_model, test_score = train_instance.train_model()
                        st.session_state.trained_model = trained_model
                        st.session_state.training_complete = True

            
            with col2:
                if train_btn:
                    st.success(f"{nav_option} Model training completed.")

                    # Display test score in two columns
                    col1, col2 = st.columns(2)  
                    st.metric(label="Test Score", value=f"{test_score:.4f}")

        else:
            st.error("The uploaded data does not meet the requirements for the selected model type.")


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
