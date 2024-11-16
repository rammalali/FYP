# app/routes/home.py
import streamlit as st
import pandas as pd
from routes.displacement import predict_displacement, plot_displacement

def render_prediction_section():
    st.subheader("Prediction")

    # Create tabs for Predict and Plot
    tab1, tab2 = st.tabs(["Predict", "Plot"])

    with tab1:
        st.write("### Predict Displacement")
        # Input fields for prediction
        start_cycle = st.number_input("Start Cycle", min_value=0, value=1, step=1)
        end_cycle = st.number_input("End Cycle", min_value=0, value=100, step=1)
        step_size = st.number_input("Step Size", min_value=1, value=10, step=1)
        velocity = st.number_input("Velocity (km/h)", min_value=1, value=60, step=1)

        if st.button("Generate Predictions"):
            if start_cycle < end_cycle:
                cycle_numbers, predicted_displacement = predict_displacement(start_cycle, end_cycle, step_size, velocity)
                results = pd.DataFrame({
                    "Cycle Number": cycle_numbers,
                    "Predicted Displacement": predicted_displacement
                })
                st.write("### Prediction Results")
                st.dataframe(results)
            else:
                st.error("Start Cycle must be less than End Cycle.")

    with tab2:
        st.write("### Plot Displacement")
        # Input fields for plotting
        start_cycle = st.number_input("Start Cycle (Plot)", min_value=0, value=1, step=1, key="plot_start_cycle")
        end_cycle = st.number_input("End Cycle (Plot)", min_value=0, value=100, step=1, key="plot_end_cycle")
        step_size = st.number_input("Step Size (Plot)", min_value=1, value=10, step=1, key="plot_step_size")
        velocity = st.number_input("Velocity (Plot) (km/h)", min_value=1, value=60, step=1, key="plot_velocity")

        if st.button("Generate Plot"):
            if start_cycle < end_cycle:
                plot_image = plot_displacement(start_cycle, end_cycle, step_size, velocity)
                st.image(plot_image, caption="Displacement Prediction Plot", use_column_width=True)
            else:
                st.error("Start Cycle must be less than End Cycle.")
