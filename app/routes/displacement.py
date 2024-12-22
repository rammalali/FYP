# app/routes/displacement.py
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import pickle
import streamlit as st


with open('models/v1/smoothed_displacement_rf_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def _predict_displacement(start_cycle, end_cycle, step_size, velocity):
    cycle_numbers = list(range(start_cycle, end_cycle + 1, step_size))
    new_data = pd.DataFrame({
        'velocity': [velocity] * len(cycle_numbers),
        'Cycle_Number': cycle_numbers
    })
    predicted_displacement = loaded_model.predict(new_data)
    return cycle_numbers, predicted_displacement.tolist()


def elastic_displacement():
    st.title("Elastic Displacement")
    st.write("Use the form below to predict displacement and view the results.")

    # Input fields
    start_cycle = st.number_input("Start Cycle:", min_value=0, value=0, step=50)
    end_cycle = st.number_input("End Cycle:", min_value=100, value=150, step=50)
    step_size = st.number_input("Step Size:", min_value=1, value=1, step=1)
    velocity = st.selectbox("Velocity (km/h):", options=[160, 210, 270, 320])

    if st.button("Predict"):
        if start_cycle < end_cycle:
            # Perform predictions
            cycle_numbers, predicted_displacement = _predict_displacement(start_cycle, end_cycle, step_size, velocity)

            # Display results in a table
            st.subheader("Predicted Displacement Table")
            result_df = pd.DataFrame({
                "Cycle Number": cycle_numbers,
                "Predicted Displacement": [f"{value:.9f}" for value in predicted_displacement]
            })
            st.dataframe(result_df)

            # Display results as a graph
            st.subheader("Predicted Displacement Graph")
            graph_image = plot_displacement_high_res(start_cycle, end_cycle, step_size, velocity)
            st.image(graph_image, caption="Predicted Displacement", use_container_width=True)
        else:
            st.error("Start Cycle must be less than End Cycle.")

def plot_displacement_high_res(start_cycle, end_cycle, step_size, velocity):
    cycle_numbers, predicted_displacement = _predict_displacement(start_cycle, end_cycle, step_size, velocity)

    plt.figure(figsize=(12, 8), dpi=300)  # Increase resolution with dpi and size
    plt.plot(cycle_numbers, predicted_displacement, label='Predicted Displacement', color='blue')
    plt.xlabel('Cycle Number')
    plt.ylabel('Displacement')
    plt.title(f'Displacement Prediction for Velocity {velocity} km/h')
    plt.grid(True)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="PNG", dpi=300)  # Save with high dpi
    plt.close()
    buf.seek(0)
    return Image.open(buf)
